# evaluate.py
import os
import json
import logging
import time
import re
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta, timezone

# --- [안정성] 모든 불필요한 라이브러리 경고 차단 ---
import warnings
warnings.filterwarnings("ignore") 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
# transformers 로깅 수준을 ERROR로 설정하여 경고 차단
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from retriever import FTCRetriever
from generator import FTCGenerator
from config import Config
from sklearn.metrics.pairwise import cosine_similarity

# 로깅 설정 (깔끔한 결과 위주)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class FTCOfficialEvaluator:
    """
    공정위 AI 공모전 공식 규격 통합 평가 시스템 (Clean & Professional Version)
    """
    def __init__(self, test_data_path: str = "eval_data.json"):
        self.test_data_path = test_data_path
        logger.info("Initializing Evaluator (Offline Mode, Warnings Suppressed)...")
        self.retriever = FTCRetriever()
        self.generator = FTCGenerator()
        
    def warmup(self):
        """실제 검색 및 생성을 1회 수행하여 모델 및 DB 캐시를 메모리에 완전히 올림"""
        logger.info("🔥 [Warmup] 실전형 전계통 예열 시작 (검색+생성 전체 사이클)...")
        dummy_q = "BGF리테일 사건에서 과징금 부과 근거는?"
        try:
            # 1. 검색 예열
            ret_result = self.retriever.retrieve(dummy_q)
            # 2. 생성 예열 (가장 관련 있는 청크 하나 사용)
            if ret_result["final_chunks"]:
                _ = self.generator.generate(dummy_q, [ret_result["final_chunks"][0]])
            logger.info("✅ [Warmup] 예열 완료. 이제 첫 문항부터 제 속도가 나옵니다.")
        except Exception as e:
            logger.error(f"⚠️ [Warmup] 예열 중 오류 발생 (무시하고 진행): {e}")
        
    def calculate_f1(self, prediction, ground_truth):
        prediction_tokens = re.findall(r'\w+', str(prediction))
        ground_truth_tokens = re.findall(r'\w+', str(ground_truth))
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def calculate_semantic_similarity(self, ans1, ans2):
        # Retriever의 dense_model(FastEmbed)을 사용하여 임베딩 및 유사도 계산
        v1 = list(self.retriever.dense_model.embed([ans1]))[0].reshape(1, -1)
        v2 = list(self.retriever.dense_model.embed([ans2]))[0].reshape(1, -1)
        return float(cosine_similarity(v1, v2)[0][0])

    def calculate_mrr(self, gt_id, retrieved_ids):
        try:
            return 1.0 / (retrieved_ids.index(gt_id) + 1)
        except ValueError:
            return 0.0

    def run_evaluation(self):
        if not os.path.exists(self.test_data_path):
            logger.error(f"Dataset not found: {self.test_data_path}")
            return

        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        # [Session Setup] 한국 시간 기준 세션 시간 생성
        kst = timezone(timedelta(hours=9))
        session_now = datetime.now(kst)
        session_time = session_now.strftime('%Y%m%d_%H%M')
        logger.info(f"📅 Evaluation session started: {session_now.strftime('%Y-%m-%d %H:%M:%S')} (KST)")

        # [Warmup] 평가 시작 전 실전 예열 수행
        self.warmup()
        
        total_count = len(test_data)
        results = []
        debug_logs = []
        logger.info(f"Evaluation started for {total_count} cases. This may take a few minutes...")

        total_item_times = []
        retrieval_times = []
        generation_times = []

        for i, item in enumerate(test_data):
            item_start_time = time.time()
            query = item['question']
            gt_id = item['chunk_id']
            gt_answer = item['answer']
            gt_doc = item.get('document_name', 'Unknown')
            
            # 1. Retrieval
            t_ret_start = time.time()
            hyde_query = self.generator.generate_hyde(query) if Config.USE_HYDE else None
            
            ret_result = self.retriever.retrieve(query, hyde_query)
            final_chunks = ret_result["final_chunks"]
            final_ids = [c['id'] for c in final_chunks]
            
            rec_5 = 1.0 if gt_id in final_ids else 0.0
            rank = (final_ids.index(gt_id) + 1) if rec_5 > 0 else "N/A"
            ret_time = time.time() - t_ret_start
            
            print(f"{'-'*80}")
            print(f" [검색 결과] Recall@5: {'✅' if rec_5 > 0 else '❌'} | 순위: {rank} | 검색시간: {ret_time:.2f}s")
            print(f"{'-'*80}")

            # 2. Generation (Using HyDE results as official)
            gen_start = time.time()
            context = "\n\n".join([f"[{c['header']}] {c['text']}" for c in final_chunks])
            gen_answer = self.generator.generate(query, final_chunks) # generator.py의 메서드명에 맞춤
            gen_time = time.time() - gen_start
            
            total_item_time = time.time() - item_start_time
            total_item_times.append(total_item_time)
            retrieval_times.append(ret_time)
            generation_times.append(gen_time)

            # 3. Score
            recall_5 = rec_5
            mrr = self.calculate_mrr(gt_id, final_ids)
            bert_sim = self.calculate_semantic_similarity(gen_answer, gt_answer)
            f1 = self.calculate_f1(gen_answer, gt_answer)
            
            final_score = (0.35 * recall_5) + (0.15 * mrr) + (0.30 * bert_sim) + (0.20 * f1)
            
            # [Debug Info] Console Output (공모전 평가 규격에 맞춘 상세 출력)
            latency_status = "✅ PASS" if total_item_time < 30 else "❌ FAIL (0점 처리)"
            
            print(f" ⏱️  [속도 평가]  전체: {total_item_time:.2f}s | 검색: {ret_time:.2f}s | 생성: {gen_time:.2f}s  -> {latency_status}")
            print(f" 🔍 [검색 평가]  Recall@5: {'✅ PASS' if recall_5 > 0 else '❌ FAIL'} | 순위(Rank): {int(1/mrr) if mrr > 0 else 'N/A'}")
            print(f" 🎯 [정답 위치]  ID: {gt_id} | 문서: {gt_doc}")
            
            # 1차 검색 진단 (데이터 규격 일치 확인 완료)
            pre_match_found = any(c['id'] == gt_id for c in ret_result["diagnostics"]["pre_rerank_candidates"])
            diag_str = "🎯 1차(Hybrid)에서 찾음" if pre_match_found else "❌ 1차에서 놓침"
            print(f" 🛠️  [검색 진단]  {diag_str}")

            print(f"\n 🏆 [상세 점수]")
            print(f"    - Recall@5 (35%): {recall_5:.2f}")
            print(f"    - MRR      (15%): {mrr:.4f}")
            print(f"    - BERTScore(30%): {bert_sim:.4f}")
            print(f"    - F1 Score (20%): {f1:.4f}")
            print(f"    👉 최종 합산 점수: {final_score*100:.2f} / 100")
            
            print(f"\n 🤖 [AI 답변]\n {gen_answer[:500]}...")
            print(f"{'='*80}")

            total_item_time = time.time() - item_start_time
            results.append({
                "no": i + 1,
                "final_score": final_score, 
                "recall_5": recall_5,
                "mrr": mrr,
                "bert_sim": bert_sim,
                "f1": f1,
                "lat_ret": ret_time,
                "lat_gen": gen_time,
                "latency": total_item_time
            })
            
            # [Debug Log] Collect for file output (모든 점수 지표 포함)
            debug_logs.append({
                "no": i + 1,
                "question": query,
                "gt_id": gt_id,
                "gt_doc": gt_doc,
                "retrieved": [{"id": c['id'], "doc": c.get('case_title', 'N/A')} for c in final_chunks],
                "pre_rerank": ret_result.get("diagnostics", {}).get("pre_rerank_candidates", []),
                "gen_answer": gen_answer,
                "gt_answer": gt_answer,
                "recall_5": recall_5,
                "mrr": mrr,
                "bert_sim": bert_sim,
                "f1": f1,
                "final_score": final_score,
                "latency": total_item_time
            })

            # [Checkpoint] 매 5문항마다 중간 저장
            if (i + 1) % 5 == 0 or (i + 1) == total_count:
                checkpoint_path = f"evaluation_checkpoint_{session_time}.json"
                df_temp = pd.DataFrame(results)
                summary_stats = df_temp.mean().to_dict()
                
                checkpoint_data = {
                    "last_updated": datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S'),
                    "overall_stats": summary_stats,
                    "progress": f"{i+1}/{total_count}",
                    "details": debug_logs
                }
                with open(checkpoint_path, "w", encoding="utf-8") as cp_f:
                    json.dump(checkpoint_data, cp_f, ensure_ascii=False, indent=4)
                logger.info(f"💾 [Checkpoint] {i+1}번째 결과 저장 완료 -> {checkpoint_path}")
                logger.info(f"   (Avg Recall - Orig: {summary_stats.get('recall_orig', 0)*100:.1f}%, HyDE: {summary_stats.get('recall_hyde', 0)*100:.1f}%)")

        # Reporting
        df = pd.DataFrame(results)
        summary = df.mean().to_dict()
        kst = timezone(timedelta(hours=9))
        now_kst_obj = datetime.now(kst)
        now_kst = now_kst_obj.strftime('%Y-%m-%d %H:%M:%S')
        time_suffix = now_kst_obj.strftime('%Y%m%d_%H%M')
        
        # 1. Official Summary Report (파일명에 시간 포함)
        report_path = f"evaluation_report_{time_suffix}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 🏆 공정위 AI 공모전 평가 결과 리포트\n\n")
            f.write(f"**평가 일시**: {now_kst} (KST)\n")
            f.write(f"**평가 대상**: {total_count} 문항\n")
            f.write(f"**최종 점수 (Final Score)**: **{summary['final_score'] * 100:.2f} / 100**\n\n")
            
            f.write("### [1] 성능 및 속도 요약\n")
            f.write(f"- **평균 응답 시간 (Avg Latency)**: **{summary['latency']:.2f} 초**\n")
            f.write(f"- **평균 생성 시간 (Avg Gen)**: {summary['lat_gen']:.2f} 초\n\n")

            f.write("### [2] 지표별 평균 점수 (Final Selection)\n")
            f.write("| 지표 | 가중치 | 평균 점수 |\n| :--- | :---: | :---: |\n")
            f.write(f"| Recall@5 | 35% | {summary['recall_5']:.4f} |\n")
            f.write(f"| MRR | 15% | {summary['mrr']:.4f} |\n")
            f.write(f"| BERTScore (Sim) | 30% | {summary['bert_sim']:.4f} |\n")
            f.write(f"| F1 Score | 20% | {summary['f1']:.4f} |\n\n")

            # [3] 검색 성능 진단
            f.write("### [3] 검색 성능 진단\n")
            f.write(f"- **최종 Recall@5**: **{summary['recall_5']*100:.2f}%**\n")
            f.write(f"- **평균 검색 시간**: {summary['lat_ret']:.2f}s\n\n")

        # 2. Detailed Debug Log (전수 조사용)
        with open("evaluation_debug_log.md", "w", encoding="utf-8") as f:
            f.write("# 🔍 평가 상세 디버그 로그 (전수 조사용)\n\n")
            f.write("| 번호 | 질문 (일부) | 정답 ID | 결과 | 검색된 ID #1 | 최종점수 |\n")
            f.write("| :--- | :--- | :--- | :---: | :--- | :---: |\n")
            for log in debug_logs:
                res_mark = "✅" if log['recall_5'] > 0 else "❌"
                short_q = log['question'][:20] + "..."
                r_ids = [r['id'] for r in log['retrieved']]
                f.write(f"| {log['no']} | {short_q} | `{log['gt_id']}` | {res_mark} | `{r_ids[0]}` | {log['final_score']:.3f} |\n")
            
            f.write("\n\n## 📝 문항별 상세 검색 내역\n")
            for log in debug_logs:
                f.write(f"\n### [{log['no']}] {log['question']}\n")
                f.write(f"- **정답**: `{log['gt_id']}` (문서: {log['gt_doc']})\n")
                
                # 최종 Top 5 (리랭킹 후)
                f.write("- **최종 검색 결과 (Top 5, After Reranking)**:\n")
                for j, r in enumerate(log['retrieved']):
                    match = " (MATCH! ✅)" if r['id'] == log['gt_id'] else ""
                    f.write(f"  {j+1}. `{r['id']}` | {r['doc']}{match}\n")
                
                # 리랭킹 전 후보군 (Top 20)
                f.write("- **리랭킹 전 후보군 (Top 20, Hybrid Search Only)**:\n")
                pre_match = False
                for j, r in enumerate(log['pre_rerank']):
                    match = " (FOUND HERE! 🎯)" if r['id'] == log['gt_id'] else ""
                    if match: pre_match = True
                    f.write(f"  {j+1}. `{r['id']}` | {r['case_title']}{match}\n")
                
                if not pre_match:
                    f.write("  ⚠️ *1차 검색(Hybrid)에서 정답을 찾지 못했습니다.*\n")
                elif log['recall_5'] == 0:
                    f.write("  ⚠️ *1차 검색에서는 찾았으나, 리랭커가 순위 밖으로 밀어냈습니다.*\n")

                f.write(f"- **생성된 답변 (AI)**: {log['gen_answer']}\n")
                f.write(f"- **실제 정답 (GT)**: {log['gt_answer']}\n")
                f.write(f"- **점수**: Recall@5={log.get('recall_5', 0)}, MRR={log.get('mrr', 0):.4f}, BERTSim={log.get('bert_sim', 0):.4f}, Final={log.get('final_score', 0):.4f}\n")

        print("\n" + "="*60)
        print(f"🏁 EVALUATION SUCCESSFUL | FINAL SCORE: {summary['final_score']*100:.2f}")
        print(f"📄 Summary: 'evaluation_report.md'")
        print(f"📄 Debug Log: 'evaluation_debug_log.md' (전수 조사용)")
        print("="*60)
        
        self.retriever.close()

if __name__ == "__main__":
    # [설정] 현재 폴더에 있는 eval_data.json으로 평가 진행
    evaluator = FTCOfficialEvaluator("eval_data.json")
    evaluator.run_evaluation()
