# main.py
import logging
from typing import Dict, Any

from config import Config
from retriever import FTCRetriever
from generator import FTCGenerator

# 로깅 설정 (config 반영)
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class FTCChatbot:
    """
    공정거래위원회 의결서 기반 RAG 챗봇 메인 클래스.
    리모델링된 Qdrant 네이티브 엔진을 사용하여 30초 이내에 답변을 생성합니다.
    """
    def __init__(self):
        # 검색기 및 생성기 초기화
        self.retriever = FTCRetriever()
        self.generator = FTCGenerator()
        logger.info("✅ [Chatbot] 모든 시스템이 준비되었습니다.")

    def answer(self, question: str) -> Dict[str, Any]:
        """질문에 대해 검색 및 생성을 수행하여 최종 결과 반환"""
        logger.info(f"❓ 질문 수신: {question[:50]}...")
        
        # 1. 검색 (Retrieve)
        # 30초 제한 사수를 위해 내부적으로 최적화된 리팩토링 버전 사용
        retrieval_res = self.retriever.retrieve(question)
        chunks = retrieval_res["final_chunks"]
        
        # 2. 생성 (Generate)
        answer = self.generator.generate(question, chunks)
        
        # 3. 결과 포맷팅 (공모전 평가 데이터 포맷 준수)
        return {
            "answer": answer,
            "retrieved_ids": [c["id"] for c in chunks], # 정확히 5개 ID 반환 보장
            "diagnostics": retrieval_res.get("diagnostics", {})
        }

if __name__ == "__main__":
    # 간단한 테스트 실행
    chatbot = FTCChatbot()
    sample_q = "BGF리테일 사건에서 공정거래위원회가 과징금을 부과하게 된 근거는?"
    result = chatbot.answer(sample_q)
    
    print("\n" + "="*50)
    print(f"Q: {sample_q}")
    print(f"A: {result['answer']}")
    print(f"IDs: {result['retrieved_ids']}")
    print("="*50)