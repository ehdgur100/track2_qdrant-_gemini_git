# config.py
import torch
import os

class Config:
    """
    RAG 시스템의 핵심 설정을 관리하는 마스터 클래스.
    대상 환경: AWS A10G (24GB VRAM) / RAM 32GB 전용 최적화 세팅
    """
    # [경로 설정]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.getenv("FTC_DATA_DIR", os.path.join(os.path.dirname(BASE_DIR), "data"))
    DB_PATH = os.path.join(BASE_DIR, "cache", "qdrant_db")
    
    # [모델 설정]
    MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    RERANK_MODEL_ID = "Dongjin-kr/ko-reranker"
    EMBED_MODEL_ID = "intfloat/multilingual-e5-large"
    SPARSE_MODEL_ID = "prithivida/Splade_PP_en_v1"

    # [하드웨어 최적화 - A10G 24GB 전용]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # BF16 정밀도로 24GB VRAM 내에서 성능과 속도의 균형을 맞춤
    USE_4BIT = False 
    TORCH_DTYPE = torch.bfloat16 

    # [검색 정확도 파라미터]
    RETRIEVAL_TOP_K = 250   # 1차 검색 후보군 확대 (정확도 향상)
    RERANK_TOP_K = 5        # 공모전 필수 규칙: 반드시 정확히 5개 반환해야 함
    USE_HYDE = False 
    USE_SPARSE = False      # 영어 Sparse(Splade)가 성능을 저하시키므로 비활성화
    
    # RRF (Reciprocal Rank Fusion) 가중치
    RRF_K = 60 # 순위 기반 병합 시의 상수값
    
    # [생성 파라미터]
    MAX_NEW_TOKENS = 120   # 30초 타임아웃 방지를 위해 최적화 (BF16 환경)
    TEMPERATURE = 0.0      # 가이드 권장사항: 일관성을 위해 0.0(Greedy Search) 설정
    TOP_P = 0.9

