# generator.py
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any

from config import Config

logger = logging.getLogger(__name__)

# 외부 라이브러리 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class FTCGenerator:
    """
    EXAONE-3.5 모델을 사용하여 의결서 기반 답변을 생성하는 엔진.
    법률적 정확성과 문장 유창성을 극대화하도록 튜닝되었습니다.
    """
    def __init__(self):
        logger.info(f"🤖 [Generator] 모델 로드 시작: {Config.MODEL_ID}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_ID, 
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 4-bit 양자화 및 bfloat16을 활용하여 VRAM 효율 및 속도 향상
        # EXAONE은 BitsAndBytesConfig를 통해 명시적으로 양자화 설정을 전달해야 합니다.
        from transformers import BitsAndBytesConfig
        quantization_config = None
        if Config.USE_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID,
            quantization_config=quantization_config,
            torch_dtype=Config.TORCH_DTYPE,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        # [Warm-up] 첫 생성 지연 방지를 위한 예열
        try:
            logger.info("🔥 [Generator] 엔진 예열 중...")
            dummy_chunks = [{"case_title": "테스트", "header": "테스트", "text": "테스트"}]
            self.generate("안녕?", dummy_chunks)
            logger.info("✅ [Generator] 준비 완료")
        except:
            pass

    def _build_prompt(self, question: str, context: str) -> str:
        """법률 전문가 persona를 부여한 시스템 프롬프트 구성"""
        system_msg = (
            "당신은 공정거래위원회 의결서를 분석하는 전문 법률 어시스턴트입니다.\n"
            "⚠️ 반드시 지켜야 할 답변 규칙:\n"
            "1. **즉시 본론 시작**: '문의하신 내용은~'과 같은 인사말이나 서론은 절대 쓰지 마십시오. 첫 문장부터 바로 핵심 정보와 근거를 작성하십시오.\n"
            "2. **정보 종합**: 제공된 여러 의결서 발췌 내용에 정보가 흩어져 있을 경우, 이를 논리적으로 종합하여 하나의 완성된 답변으로 작성하십시오.\n"
            "3. **정확한 수치**: 과징금액, 기간, 날짜 등 수치 정보는 의결서에 적힌 그대로 정확하게 전달하십시오.\n"
            "4. **경어체 및 구조화**: 정중한 경어체(~습니다, ~입니다)를 사용하고, 내용이 복잡할 경우 불렛 포인트(•)로 요약하십시오."
        )
        
        # EXAONE-3.5의 프롬프트 템플릿 준수
        prompt = f"[인스트럭션]\n{system_msg}\n\n[의결서 내용]\n{context}\n\n[질문]\n{question}\n\n[답변]\n"
        return prompt

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """검색된 청크를 결합하여 최종 답변 생성"""
        if not retrieved_chunks:
            return "제공된 의결서 내용에서 관련 정보를 찾을 수 없습니다."

        # 맥락 구성 방식 변경: '청크'라는 단어를 빼고 중립적인 구분선 사용
        context = "\n\n".join([
            f"[의결서 발췌내용: {c['header']}]\n{c['text']}" 
            for i, c in enumerate(retrieved_chunks)
        ])

        prompt = self._build_prompt(question, context)
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "max_new_tokens": Config.MAX_NEW_TOKENS,
            "max_length": None, # transformers 경고 방지
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False # 법률 답변의 일관성을 위해 Greedy Search 사용
        }

        try:
            outputs = self.pipe(
                prompt,
                **gen_kwargs,
                return_full_text=False
            )
            answer = outputs[0]["generated_text"].strip()
            return answer
        except Exception as e:
            logger.error(f"❌ [Generator] 답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."

    def generate_hyde(self, question: str) -> str:
        """HyDE(Hypothetical Document Embeddings)용 가상 답변 생성"""
        # HyDE 프롬프트 최적화: 공정위 의결서 특유의 문체와 법률 용어를 유도
        hyde_prompt = (
            f"[인스트럭션]\n대한민국 공정거래위원회 조사관으로서 다음 질문에 대한 가상의 의결서 답변을 작성하세요. "
            f"사건의 핵심 위반 행위, 근거 법령(예: 하도급법 제4조), 위원회의 판단 근거를 포함하여 전문적인 문체로 작성하십시오.\n\n"
            f"[질문]\n{question}\n\n[답변]\n"
        )
        try:
            outputs = self.pipe(
                hyde_prompt,
                max_new_tokens=150,
                max_length=None, # transformers 경고 방지
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            hyde_answer = outputs[0]["generated_text"].strip()
            logger.info(f"💡 [HyDE] 생성 완료: {hyde_answer[:50]}...")
            return hyde_answer
        except Exception as e:
            logger.error(f"❌ [Generator] HyDE 생성 실패: {e}")
            return question # 실패 시 원본 질문 반환
