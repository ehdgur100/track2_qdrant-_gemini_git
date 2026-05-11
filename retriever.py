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

# 외부 라이브러리 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class FTCRetriever:
    """
    Qdrant 네이티브 하이브리드 검색 엔진.
    30초 응답 시간 규정을 준수하면서 Recall@5를 극대화하도록 설계되었습니다.
    """
    def __init__(self):
        logger.info("🔍 [Retriever] Qdrant 네이티브 엔진 초기화 시작...")
        
        self.client = QdrantClient(path=Config.DB_PATH)
        self.collection_name = "ftc_chunks_native"
        
        self.dense_model = TextEmbedding(model_name=Config.EMBED_MODEL_ID)
        if Config.USE_SPARSE:
            self.sparse_model = SparseTextEmbedding(model_name=Config.SPARSE_MODEL_ID)
        
        self.rerank_model = CrossEncoder(
            Config.RERANK_MODEL_ID, 
            device=Config.DEVICE,
            local_files_only=True
        )
        
        # 사건명 목록 캐싱
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
                    if title: self.all_case_titles.add(title)
                if next_page_offset is None: break
            
            self.all_case_titles = sorted(list(self.all_case_titles), key=len, reverse=True)
            logger.info(f"🏷️ [Retriever] {len(self.all_case_titles)}개의 사건명 로드 및 정렬 완료")
        except Exception as e:
            logger.error(f"❌ [Retriever] 사건명 로드 실패: {e}")
            self.all_case_titles =[]

        try:
            logger.info("🔥 [Retriever] 엔진 예열 중...")
            self.retrieve("테스트 질문")
            logger.info("✅ [Retriever] 준비 완료")
        except:
            pass

    def _get_normalized_case(self, text: str) -> str:
        """사건명 비교를 위한 기본 정규화 (동의어 처리 제거)"""
        if not text: return ""
        text = re.sub(r'\(주\)|\(사\)', '', text)
        return re.sub(r'\s+', '', text).strip().upper()

    def retrieve(self, question: str, hyde_query: str = None) -> Dict[str, Any]:
        """검색 수행 (단일 하드 필터링 적용)"""
        search_query = hyde_query if hyde_query else question
        
        # 1. 사건명 감지
        detected_case = None
        norm_question = self._get_normalized_case(question)
        for title in self.all_case_titles:
            norm_title = self._get_normalized_case(title)
            if norm_title and norm_title in norm_question:
                detected_case = title
                break
        
        qdrant_filter = None
        if detected_case:
            qdrant_filter = models.Filter(
                must=[models.FieldCondition(key="case_title", match=models.MatchValue(value=detected_case))]
            )
            logger.info(f"🎯 [Retriever] 하드 필터링 적용: {detected_case}")

        # 🚨 e5-large 규칙: 검색 시 'query: ' 접두사 추가 필수
        # 가장 깔끔하고 효율적인 방식
        query_dense = next(self.dense_model.embed([f"query: {search_query}"])).tolist()
        
        try:
            # 하드 필터링을 유지하여 타 사건 문서의 노이즈 원천 차단
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense,
                using="dense",
                query_filter=qdrant_filter,
                limit=Config.RETRIEVAL_TOP_K
            )
            
            candidates =[
                {
                    "id": p.payload.get("id"),
                    "text": p.payload.get("text", ""),
                    "case_title": p.payload.get("case_title", ""),
                    "header": p.payload.get("header", ""),
                    "enriched_text": p.payload.get("enriched_text", p.payload.get("text", "")),
                    "chunk_id": p.payload.get("chunk_id", "")
                }
                for p in res.points
            ]
        except Exception as e:
            logger.error(f"❌ [Retriever] 검색 실패: {e}")
            candidates =[]

        if not candidates:
            return {"final_chunks":[], "diagnostics": {"detected_case": detected_case, "pre_rerank_candidates": []}}

        passages = [f"[사건: {c['case_title']}] ({c['header']}) {c['enriched_text']}" for c in candidates]
        
        import torch
        torch.cuda.empty_cache()
        rerank_scores = self.rerank_model.predict(
            [(question, p) for p in passages],
            batch_size=16, 
            show_progress_bar=False
        )
        torch.cuda.empty_cache()
        
        # 4. Final Scoring with Heuristics (룰베이스 보너스 총집합)
        for i, score in enumerate(rerank_scores):
            cand = candidates[i]
            final_score = float(score)
            text = cand.get("text", "")
            header = cand.get("header", "")
            
            # [1] 사건명 일치 강력 보너스
            if detected_case and detected_case in cand["case_title"]:
                final_score += 1.0
            
            # [2] 키워드 매칭 보너스 (Sparse 보완)
            query_words = set([w for w in re.findall(r'\w+', question) if len(w) >= 2])
            text_words = set(re.findall(r'\w+', text))
            overlap_count = len(query_words & text_words)
            final_score += min(overlap_count * 0.05, 0.5)

            # [3] 법 조항 매칭
            law_articles = re.findall(r'제\d+조', question)
            for article in law_articles:
                if article in text:
                    final_score += 0.5

            # [4] 과징금/금액 관련 질문
            if any(q in question for q in["과징금", "금액", "얼마", "납부"]):
                if any(h in header for h in["주 문", "주문", "결 론", "결론", "과징금"]):
                    final_score += 0.3
                if re.search(r'\d+(?:,\d+)*\s*원', text):
                    final_score += 0.2

            # [5] 날짜/기간 관련 질문
            if any(q in question for q in["언제", "기간", "일자", "날짜", "년", "월", "일"]):
                if re.search(r'\d{4}[\.년]\s*\d{1,2}[\.월]', text):
                    final_score += 0.2

            # [6] 위반 행위/사실 확인 관련 질문 (회원님 아이디어 적용!)
            if any(q in question for q in["위반", "내용", "사실", "행위", "사유", "근거"]):
                target_headers =["사실의 확인", "사실의확인", "위반행위", "위반 행위", "행위사실", "행위 사실", "위법성 판단", "사실의 인정"]
                if any(h in header for h in target_headers):
                    final_score += 0.2

            # [7] 초반부 청크 타이브레이커
            chunk_id = cand.get("id", "")
            try:
                if int(chunk_id.split('-CH-')[-1]) <= 5:
                    final_score += 0.3
            except:
                pass
            
            cand["final_score"] = final_score
            
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 5개 추출
        final_chunks =[]
        seen_ids = set()
        
        for cand in candidates:
            c_id = cand.get("id")
            if c_id and c_id not in seen_ids:
                final_chunks.append(cand)
                seen_ids.add(c_id)
            if len(final_chunks) >= Config.RERANK_TOP_K:
                break
        
        #[안전장치 1] candidates 내부에서 보충
        if len(final_chunks) < Config.RERANK_TOP_K:
            logger.warning(f"⚠️ [Retriever] 결과 부족({len(final_chunks)}개). 내부 보충 시도.")
            for cand in candidates:
                c_id = cand.get("id")
                if c_id not in seen_ids:
                    final_chunks.append(cand)
                    seen_ids.add(c_id)
                if len(final_chunks) >= Config.RERANK_TOP_K:
                    break

        top_results = final_chunks[:Config.RERANK_TOP_K]

        # 🚨 [최후의 보루 절대 방어막] (0점 방지)
        if len(top_results) < Config.RERANK_TOP_K:
            logger.critical(f"🆘 [Retriever] 청크 부족! 0점 방지를 위해 강제로 채웁니다.")
            try:
                fallback_res = self.client.scroll(collection_name=self.collection_name, limit=15, with_payload=True, with_vectors=False)[0]
                for f_point in fallback_res:
                    f_id = f_point.payload.get("id")
                    if f_id and f_id not in seen_ids:
                        top_results.append({
                            "id": f_id, "text": f_point.payload.get("text", ""), "header": "", "case_title": "Fallback", "final_score": -99.0
                        })
                        seen_ids.add(f_id)
                    if len(top_results) >= Config.RERANK_TOP_K:
                        break
            except Exception as e:
                logger.error(f"Fallback 실패: {e}")

        logger.info(f"🔍 [Retriever] Final Top {len(top_results)} Chunk IDs:")
        for res in top_results:
            logger.info(f"   - {res.get('id')}: {res.get('final_score', 0):.4f}")
        
        return {
            "final_chunks": top_results,
            "diagnostics": {
                "detected_case": detected_case,
                "pre_rerank_candidates": candidates,
                "top_rerank_scores":[round(c.get("final_score", 0), 4) for c in top_results]
            }
        }

    def close(self):
        try:
            self.client.close()
            logger.info("🔒 [Retriever] Qdrant 연결 종료")
        except:
            pass