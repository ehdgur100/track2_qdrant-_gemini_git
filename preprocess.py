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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class FTCPreprocessor:
    def __init__(self):
        logger.info(f"⚙️[Init] Qdrant 및 임베딩 모델 로딩 중... (Device: {Config.DEVICE})")
        
        os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
        self.client = QdrantClient(path=Config.DB_PATH)
        
        self.dense_model = TextEmbedding(model_name=Config.EMBED_MODEL_ID)
        if Config.USE_SPARSE:
            self.sparse_model = SparseTextEmbedding(model_name=Config.SPARSE_MODEL_ID)
        
        self.collection_name = "ftc_chunks_native"

    def create_indices(self):
        loader = FTCDataLoader(Config.DATA_DIR)
        chunks = loader.load_and_enrich()
        
        if not chunks:
            logger.error("❌ 처리할 청크 데이터가 없습니다.")
            return

        logger.info(f"⚡ [Preprocess] 컬렉션 '{self.collection_name}' 초기화 중...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            }
        )

        logger.info("⚡ [Preprocess] 하이브리드 벡터 인덱싱 시작...")
        
        batch_size = 64
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
            batch = chunks[i : i + batch_size]
            
            # 🚨 e5-large 규칙: 인덱싱 시 'passage: ' 추가 필수
            enriched_texts = [f"passage: {c['enriched_text']}" for c in batch]
            
            dense_vectors = list(self.dense_model.embed(enriched_texts))
            
            points =[]
            for j, (chunk, d_vec) in enumerate(zip(batch, dense_vectors)):
                points.append(
                    models.PointStruct(
                        id=i + j,
                        vector={"dense": d_vec.tolist()},
                        payload={
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "case_title": chunk["case_title"],
                            "header": chunk["header"],
                            "type": chunk["type"],
                            "enriched_text": chunk["enriched_text"]
                        }
                    )
                )
            
            self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"✅ [Preprocess] 인덱싱 완료: 총 {len(chunks)}개 청크가 저장되었습니다.")

if __name__ == "__main__":
    preprocessor = FTCPreprocessor()
    preprocessor.create_indices()