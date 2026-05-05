# retriever.py
import logging
import re
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

from config import Config

logger = logging.getLogger(__name__)

class FTCRetriever:
    """
    Qdrant 네이티브 하이브리드 검색 엔진.
    30초 응답 시간 규정을 준수하면서 Recall@5를 극대화하도록 설계되었습니다.
    """
    def __init__(self):
        logger.info("🔍 [Retriever] Qdrant 네이티브 엔진 초기화 시작...")
        
        # 1. 클라이언트 및 모델 로드
        self.client = QdrantClient(path=Config.DB_PATH)
        self.collection_name = "ftc_chunks_native"
        
        # 임베딩 모델 (기본 디바이스 할당)
        self.dense_model = TextEmbedding(model_name=Config.EMBED_MODEL_ID)
        self.sparse_model = SparseTextEmbedding(model_name=Config.SPARSE_MODEL_ID)
        
        # 리랭커 로드
        self.rerank_model = CrossEncoder(
            Config.RERANK_MODEL_ID, 
            device=Config.DEVICE,
            local_files_only=True
        )
        
        # 2. 사건명 목록 캐싱 (필터링용)
        # Qdrant에서 유니크한 사건명 목록을 가져옵니다. (전체 청크 스캔)
        try:
            self.all_case_titles = set()
            next_page_offset = None
            
            while True:
                scroll_res = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000, 
                    with_payload=["case_title"],
                    with_vectors=False,
                    offset=next_page_offset
                )
                points, next_page_offset = scroll_res
                for p in points:
                    title = p.payload.get("case_title")
                    if title:
                        self.all_case_titles.add(title)
                
                if next_page_offset is None:
                    break
            
            self.all_case_titles = list(self.all_case_titles)
            logger.info(f"🏷️ [Retriever] {len(self.all_case_titles)}개의 사건명 로드 완료 (500개 전수 확인 완료)")
        except Exception as e:
            logger.error(f"❌ [Retriever] 사건명 로드 실패: {e}")
            self.all_case_titles = []

        # [Warm-up] 첫 검색 지연 방지를 위한 예열
        try:
            logger.info("🔥 [Retriever] 엔진 예열 중...")
            self.retrieve("테스트 질문")
            logger.info("✅ [Retriever] 준비 완료")
        except:
            pass

    def _get_normalized_case(self, text: str) -> str:
        """사건명 비교를 위한 정규화"""
        if not text: return ""
        text = re.sub(r'\(주\)|\(사\)', '', text)
        return re.sub(r'\s+', '', text).strip()

    def retrieve(self, question: str, hyde_query: str = None) -> Dict[str, Any]:
        """Qdrant 네이티브 하이브리드 검색 수행"""
        search_query = hyde_query if hyde_query else question
        
        # 1. 사건명 감지 (Filtering)
        detected_case = None
        norm_question = self._get_normalized_case(question)
        for title in self.all_case_titles:
            norm_title = self._get_normalized_case(title)
            if norm_title and (norm_title in norm_question or norm_question in norm_title):
                detected_case = title
                break
        
        qdrant_filter = None
        if detected_case:
            qdrant_filter = models.Filter(
                must=[models.FieldCondition(key="case_title", match=models.MatchValue(value=detected_case))]
            )
            logger.info(f"🎯 [Retriever] 사건명 필터 적용: {detected_case}")

        # 2. Hybrid Search (Dense + Sparse)
        # FastEmbed를 사용해 쿼리 벡터 생성
        query_dense = list(self.dense_model.embed([search_query]))[0].tolist()
        query_sparse_res = list(self.sparse_model.embed([search_query]))[0]
        query_sparse = models.SparseVector(
            indices=query_sparse_res.indices.tolist(),
            values=query_sparse_res.values.tolist()
        )

        try:
            # Qdrant v1.10+의 최신 기능을 사용해 하이브리드 검색 수행 (RRF 병합)
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=query_dense, using="dense", filter=qdrant_filter, limit=Config.RETRIEVAL_TOP_K),
                    models.Prefetch(query=query_sparse, using="sparse", filter=qdrant_filter, limit=Config.RETRIEVAL_TOP_K),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=Config.RETRIEVAL_TOP_K
            )
            
            # [안전장치] 필터를 적용했는데 결과가 너무 적으면 필터 없이 재검색
            if len(response.points) < 5 and qdrant_filter is not None:
                logger.warning(f"⚠️ [Retriever] 필터 결과 부족({len(response.points)}개). 필터 없이 재검색 수행.")
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(query=query_dense, using="dense", limit=Config.RETRIEVAL_TOP_K),
                        models.Prefetch(query=query_sparse, using="sparse", limit=Config.RETRIEVAL_TOP_K),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=Config.RETRIEVAL_TOP_K
                )

            candidates = [
                {
                    "id": p.payload["id"],
                    "text": p.payload["text"],
                    "case_title": p.payload["case_title"],
                    "header": p.payload["header"],
                    "enriched_text": p.payload.get("enriched_text", p.payload["text"]),
                "chunk_id": p.payload.get("chunk_id", "")
            }
            for p in response.points
        ]
        except Exception as e:
            logger.error(f"❌ [Retriever] 하이브리드 검색 실패: {e}")
            return {
                "final_chunks": [], 
                "diagnostics": {
                    "detected_case": detected_case,
                    "pre_rerank_candidates": [],
                    "error": str(e)
                }
            }

        if not candidates:
            return {
                "final_chunks": [], 
                "diagnostics": {
                    "detected_case": detected_case,
                    "pre_rerank_candidates": []
                }
            }

        # 리랭킹은 컨텍스트가 포함된 enriched_text를 사용합니다. (OOM 방지를 위해 배치 사이즈 지정)
        passages = [f"[사건: {c['case_title']}] ({c['header']}) {c['enriched_text']}" for c in candidates]
        # [VRAM 최적화] 리랭킹 전 캐시 정리 및 배치 사이즈 최소화
        import torch
        torch.cuda.empty_cache()
        
        rerank_scores = self.rerank_model.predict(
            [(question, p) for p in passages],
            batch_size=16,      # A10G 24GB 환경에 맞춰 16으로 상향 (속도 개선)
            show_progress_bar=False
        )
        torch.cuda.empty_cache()
        
        for i, score in enumerate(rerank_scores):
            cand = candidates[i]
            final_score = float(score)
            
            # [도메인 최적화 1] 사건명 일치 보너스 (+1.0)
            if detected_case and detected_case in cand["case_title"]:
                final_score += 1.0
            
            # [도메인 최적화 2] 핵심 섹션 보너스 (+0.2)
            # 너무 높으면(예: 1.5) 다른 사건의 주문이 섞여 들어오는 부작용(환각)이 발생함
            header = cand.get("header", "")
            if any(h in header for h in ["주 문", "주문", "결 론", "결론"]):
                final_score += 0.2

            # [도메인 최적화 3] 초반부 청크 타이브레이커 (+0.3)
            chunk_id = cand.get("id", "") # payload의 'id' 사용
            try:
                chunk_num = int(chunk_id.split('-CH-')[-1]) if '-CH-' in chunk_id else 999
                if chunk_num <= 5:
                    final_score += 0.3
            except:
                pass
            
            cand["final_score"] = final_score
            
        # 5. 가공된 점수로 재정렬 및 결과 반환
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # [공모전 필수 규칙] 중복 없는 정확히 5개의 chunk_id 추출
        final_chunks = []
        seen_ids = set()
        
        for cand in candidates:
            c_id = cand.get("id") # 원본 chunk_id
            if c_id and c_id not in seen_ids:
                final_chunks.append(cand)
                seen_ids.add(c_id)
            if len(final_chunks) >= Config.RERANK_TOP_K:
                break
        
        # [안전장치] 만약 5개가 안 될 경우 (거의 없겠지만) candidates 전체에서 보충
        if len(final_chunks) < Config.RERANK_TOP_K:
            logger.warning(f"⚠️ [Retriever] 결과 부족({len(final_chunks)}개). 보충 시도.")
            for cand in candidates:
                c_id = cand.get("id")
                if c_id not in seen_ids:
                    final_chunks.append(cand)
                    seen_ids.add(c_id)
                if len(final_chunks) >= Config.RERANK_TOP_K:
                    break

        top_results = final_chunks[:Config.RERANK_TOP_K]
        
        # [로그] 상위 결과 점수 출력 (분석용)
        logger.info(f"🔍 [Retriever] Final Top {len(top_results)} Chunk IDs:")
        for res in top_results:
            logger.info(f"   - {res.get('id')}: {res.get('final_score', 0):.4f}")
        
        return {
            "final_chunks": top_results,
            "diagnostics": {
                "detected_case": detected_case,
                "pre_rerank_candidates": candidates,
                "top_rerank_scores": [round(c.get("final_score", 0), 4) for c in top_results]
            }
        }


    def close(self):
        """DB 연결 종료"""
        try:
            self.client.close()
            logger.info("🔒 [Retriever] Qdrant 연결 종료")
        except:
            pass
