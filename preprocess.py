# preprocess.py
import os
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding, SparseTextEmbedding
from tqdm import tqdm

from config import Config
from data_loader import FTCDataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class FTCPreprocessor:
    """
    Qdrant 네이티브 하이브리드 검색(Dense + Sparse)을 위한 인덱싱 엔진.
    모든 데이터와 벡터를 Qdrant 내부에서 통합 관리합니다.
    """
    def __init__(self):
        logger.info(f"⚙️ [Init] Qdrant 및 임베딩 모델 로딩 중... (Device: {Config.DEVICE})")
        
        # 1. Qdrant 로컬 클라이언트 초기화
        os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
        self.client = QdrantClient(path=Config.DB_PATH)
        
        # 2. 임베딩 모델 초기화 (FastEmbed 사용으로 속도와 메모리 효율 극대화)
        # Dense Embedding (BAAI/bge-m3)
        self.dense_model = TextEmbedding(model_name=Config.EMBED_MODEL_ID)
        
        # Sparse Embedding (SPLADE 기반 한국어 모델)
        self.sparse_model = SparseTextEmbedding(model_name=Config.SPARSE_MODEL_ID)
        
        self.collection_name = "ftc_chunks_native"

    def create_indices(self):
        """데이터를 전처리하고 Qdrant 통합 하이브리드 인덱스를 생성합니다."""
        loader = FTCDataLoader(Config.DATA_DIR)
        chunks = loader.load_and_enrich()
        
        if not chunks:
            logger.error("❌ 처리할 청크 데이터가 없습니다.")
            return

        # 1. 컬렉션 재생성 (Dense + Sparse 멀티 벡터 설정)
        logger.info(f"⚡ [Preprocess] 컬렉션 '{self.collection_name}' 초기화 중...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1024, # bge-m3 default size
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True # 메모리 절약을 위해 디스크 인덱스 사용
                    )
                )
            }
        )

        # 2. 하이브리드 인덱싱 수행
        logger.info("⚡ [Preprocess] 하이브리드 벡터 인덱싱 시작...")
        
        batch_size = 64 # A10G GPU 가속을 위한 최적화된 배치 사이즈
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
            batch = chunks[i : i + batch_size]
            enriched_texts = [c['enriched_text'] for c in batch]
            
            # 벡터 생성
            dense_vectors = list(self.dense_model.embed(enriched_texts))
            sparse_vectors = list(self.sparse_model.embed(enriched_texts))
            
            points = []
            for j, (chunk, d_vec, s_vec) in enumerate(zip(batch, dense_vectors, sparse_vectors)):
                points.append(
                    models.PointStruct(
                        id=i + j,
                        vector={
                            "dense": d_vec.tolist(),
                            "sparse": models.SparseVector(
                                indices=s_vec.indices.tolist(),
                                values=s_vec.values.tolist()
                            )
                        },
                        payload={
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "case_title": chunk["case_title"],
                            "header": chunk["header"],
                            "type": chunk["type"],
                            "enriched_text": chunk["enriched_text"] # 리랭킹 시 활용
                        }
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        logger.info(f"✅ [Preprocess] 인덱싱 완료: 총 {len(chunks)}개 청크가 Qdrant에 저장되었습니다.")

if __name__ == "__main__":
    preprocessor = FTCPreprocessor()
    preprocessor.create_indices()
