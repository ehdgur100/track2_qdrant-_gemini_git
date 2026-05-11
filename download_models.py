# download_models.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from fastembed import TextEmbedding
from config import Config

def download_all_models():
    print("🚀 [오프라인 빌드] 모든 모델 사전 다운로드 시작...")
    TextEmbedding(model_name=Config.EMBED_MODEL_ID)
    CrossEncoder(Config.RERANK_MODEL_ID)
    
    # 👇 여기에 trust_remote_code=True 옵션을 추가했습니다!
    AutoTokenizer.from_pretrained(Config.MODEL_ID, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(Config.MODEL_ID, trust_remote_code=True)
    
    print("✅ 모든 모델 다운로드 완료! 오프라인 실행 준비 끝.")

if __name__ == "__main__":
    download_all_models()