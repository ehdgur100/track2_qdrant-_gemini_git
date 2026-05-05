# 1. 베이스 이미지 설정 (CUDA 12.1 및 Python 3.10 포함)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 생성
WORKDIR /app

# 4. 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 모델 사전 다운로드 (인터넷 연결 필요)
# 빌드 시점에 모델을 다운로드하여 이미지 내부에 저장합니다.
RUN mkdir -p /app/models
RUN python3 download_models.py

# 7. 환경 변수 설정
ENV PYTHONIOENCODING=utf-8
ENV FTC_DATA_DIR=/data
ENV CUDA_VISIBLE_DEVICES=0

# 8. 최종 실행 명령어
CMD ["python3", "main.py"]
