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
    """
    def __init__(self):
        logger.info(f"🤖 [Generator] 모델 로드 시작: {Config.MODEL_ID}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_ID, 
            local_files_only=True,
            trust_remote_code=True
        )
        
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
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        try:
            logger.info("🔥 [Generator] 엔진 예열 중...")
            dummy_chunks =[{"case_title": "테스트", "header": "테스트", "text": "테스트"}]
            self.generate("안녕?", dummy_chunks)
            logger.info("✅ [Generator] 준비 완료")
        except:
            pass

    def _build_prompt(self, question: str, context: str) -> str:
        """EXAONE 공식 Chat Template 적용 (F1 Score 및 추출 성능 최적화)"""
        system_msg = (
            "당신은 공정거래위원회 의결서 전문 분석 및 정보 추출 AI입니다.\n"
            "반드시 제공된 [의결서]만을 기반으로 질문에 답변해야 하며, 다음 [답변 규칙]을 엄격하게 준수하십시오.\n\n"
            "[답변 규칙 - 핵심 추출 및 간결성]\n"
            "1. (원문 발췌) 의결서 본문의 '단어', '수치'를 그대로 사용하되, 질문에 대한 직접적인 정답 위주로 간결하게 답변하십시오.\n"
            "2. (핵심 요약) 인사말 없이 오직 정답만 개조식(bullet point)으로 출력하십시오. 불필요한 부연 설명이나 중복된 서술은 절대 금지합니다.\n"
            "3. (필수 정보 포함) 답변에 반드시 필요한 핵심 판단 근거와 수치는 누락하지 마십시오. 단, 서술은 최소한으로 유지하십시오.\n"
            "4. (수치 및 법령) 수치와 법령 기호는 원문 형태(예: '1,234,567,890원', '제x조 제x항')를 그대로 유지하십시오.\n"
            "5. (환각 방지) 제공된 의결서 내용 전체를 꼼꼼히 검토하십시오. 질문에 대한 직접적인 정답이나 관련 수치가 아주 미세하게라도 언급되어 있다면 반드시 발췌하여 답변하십시오. 만약 문서 전체를 샅샅이 뒤졌음에도 불구하고 답변을 위한 정보가 절대적으로 부족한 경우에만 \"정보 없음\"이라고 답변하십시오."
        )
        
        user_msg = f"[의결서]\n{context}\n\n[질문]\n{question}"
        
        messages =[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "제공된 의결서 내용에서 관련 정보를 찾을 수 없습니다."

        context = "\n\n".join([
            f"[사건명: {c.get('case_title', '알수없음')}] [문서위치: {c.get('header', '')}]\n내용: {c.get('text', '')}" 
            for i, c in enumerate(retrieved_chunks)
        ])

        prompt = self._build_prompt(question, context)
        
        gen_kwargs = {
            "max_new_tokens": Config.MAX_NEW_TOKENS,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False # 일관성을 위해 Greedy Search 사용
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
        """HyDE(Hypothetical Document Embeddings)용 가상 답변 생성 (원본 유지)"""
        hyde_prompt = (
            f"[인스트럭션]\n대한민국 공정거래위원회 조사관으로서 다음 질문에 대한 가상의 의결서 답변을 작성하세요. "
            f"사건의 핵심 위반 행위, 근거 법령(예: 하도급법 제4조), 위원회의 판단 근거를 포함하여 전문적인 문체로 작성하십시오.\n\n"
            f"[질문]\n{question}\n\n[답변]\n"
        )
        try:
            outputs = self.pipe(
                hyde_prompt,
                max_new_tokens=150,
                max_length=None,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            hyde_answer = outputs[0]["generated_text"].strip()
            return hyde_answer
        except Exception as e:
            return question