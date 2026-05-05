import os
from fastembed import TextEmbedding, SparseTextEmbedding
from config import Config

def download_new_models():
    print("🚀 [Build] 새로운 임베딩 모델 다운로드 시작...")

    # 1. Dense 임베딩 모델 (intfloat/multilingual-e5-large)
    print(f"📦 Downloading Dense Embedding: {Config.EMBED_MODEL_ID}")
    TextEmbedding(model_name=Config.EMBED_MODEL_ID)
    
    # 2. Sparse 임베딩 모델 (prithivida/Splade_PP_en_v1)
    print(f"📦 Downloading Sparse Embedding: {Config.SPARSE_MODEL_ID}")
    SparseTextEmbedding(model_name=Config.SPARSE_MODEL_ID)

    print("✅ [Build] 새로운 모델 다운로드 완료! (LLM은 기존 캐시 사용)")

if __name__ == "__main__":
    download_new_models()
