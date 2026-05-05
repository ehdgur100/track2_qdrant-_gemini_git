# data_loader.py
import json
import os
import re
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FTCDataLoader:
    """
    공정거래위원회 의결서 특화 초정밀 전처리 엔진.
    노이즈 제거, 중복 문단 소거, 텍스트 정규화를 통해 고순도 인덱싱용 데이터를 생성합니다.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def clean_text(self, text: str) -> str:
        """
        의결서 특유의 노이즈를 제거하되, 문서 구조(줄바꿈)는 보존합니다.
        """
        if not text: return ""
        
        # 1. 페이지 번호 제거
        text = re.sub(r'-\s?\d+\s?-', '', text)
        
        # 2. 각주 번호 제거
        text = re.sub(r'\d+\)', '', text)
        
        # 3. 한자 제거 및 한글 병기 처리
        text = re.sub(r'\([\u4e00-\u9fff]+\)', '', text)
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # 4. 불필요한 연속 공백 정규화 (줄바꿈은 유지)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        
        # 5. 빈 줄이 너무 많은 경우 정리
        text = '\n'.join([line for line in lines if line])
        
        return text.strip()

    def load_and_enrich(self) -> list:
        """
        데이터를 로드하고 검색 및 생성에 최적화된 형태로 보강합니다.
        """
        all_chunks = []
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and not f.endswith('_metadata.json')]
        
        logger.info(f"📂 [DataLoader] 총 {len(files)}개의 의결서 파일 분석 시작...")

        for file in tqdm(files, desc="Processing JSON files"):
            file_path = os.path.join(self.data_dir, file)
            meta_path = file_path.replace('_hybrid.json', '_metadata.json')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    doc_metadata = json.load(f)
            
            case_title = doc_metadata.get('case_name')
            if not case_title:
                clean_name = file.replace('_hybrid.json', '').replace('(주)', '').replace('(사)', '').strip()
                case_title = clean_name.split('의')[0].strip() if '의' in clean_name else clean_name
            
            for chunk in data:
                raw_text = chunk.get('page_content', '')
                cleaned_text = self.clean_text(raw_text)
                
                metadata = chunk.get('metadata', {})
                header = metadata.get('Header', metadata.get('header', '내용'))
                chunk_type = metadata.get('chunk_type', 'text')
                
                # 표(table) 타입인 경우 텍스트 앞에 표시를 남겨 LLM이 인지하게 함
                type_prefix = "[표] " if chunk_type == 'table' else ""
                
                # 검색 및 리랭킹용 보강 텍스트
                boosted_title = f"{case_title} {case_title}"
                enriched_text = f"[사건: {case_title}] {boosted_title} ({header}) {type_prefix}{cleaned_text}"
                
                all_chunks.append({
                    "id": metadata.get('chunk_id'),
                    "text": cleaned_text,
                    "enriched_text": enriched_text,
                    "case_title": case_title,
                    "header": header,
                    "type": chunk_type
                })
        
        logger.info(f"✅ [DataLoader] 전처리 완료. 총 {len(all_chunks)}개 청크 확보.")
        return all_chunks
