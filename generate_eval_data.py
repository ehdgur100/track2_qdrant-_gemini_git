# generate_eval_data.py
import json
import random
import os
import time
import glob
from openai import OpenAI
from config import Config
from data_loader import DataProcessor

# 🔑 OpenAI API 키 세팅 (환경 변수 또는 직접 입력)
os.environ["OPENAI_API_KEY"] = "sk-..." # 여기에 키를 입력하세요

client = OpenAI()

def generate_qa_pair(chunk_text, chunk_id):
    system_prompt = """
당신은 공정거래위원회 의결서 RAG 대회의 출제 위원입니다.
주어진 문서를 읽고, 대회 평가에 적합한 구체적인 '단답형 Q&A'를 1개 만들어주세요.

[출제 원칙]
1. 구체성: '누가, 언제, 어떤 위반을 했는지' 명확히 포함하세요.
2. 사실기반: 문서에 명시된 팩트(금액, 날짜, 조치내용 등)만 물어보세요.
3. 정답: 채점이 용이하도록 아주 짧은 단답형으로 작성하세요.

[출력 형식 (JSON)]
{
    "question": "사건명이나 기업명이 포함된 구체적 질문",
    "ground_truth_answer": "단답형 정답",
    "ground_truth_chunk_id": "제공된 chunk_id"
}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Chunk ID: {chunk_id}\n\n문서내용:\n{chunk_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

def main():
    cfg = Config()
    dp = DataProcessor()
    
    print("📂 원본 데이터 분석 중...")
    corpus = dp.load_all(cfg.DATA_DIR)
    
    if not corpus:
        print("❌ 데이터를 찾을 수 없습니다. data/ 폴더를 확인해주세요.")
        return

    # 의미 있는 텍스트 위주로 샘플링
    valid_chunks = [c for c in corpus if len(c['text']) > 100]
    sample_chunks = random.sample(valid_chunks, min(100, len(valid_chunks)))

    eval_set = []
    print(f"🤖 총 {len(sample_chunks)}개의 청크에서 문제를 출제합니다...")

    for chunk in tqdm(sample_chunks):
        qa = generate_qa_pair(chunk['enriched_text'], chunk['id'])
        if qa and qa.get('question'):
            eval_set.append(qa)
            if len(eval_set) >= 30: # 예시로 30문제만 생성
                break
        time.sleep(0.5)

    with open("eval_data.json", "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 평가 데이터셋 생성 완료! (eval_data.json, 총 {len(eval_set)}문제)")

if __name__ == "__main__":
    from tqdm import tqdm
    main()
