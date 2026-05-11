FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN conda run pip install --force-reinstall transformers==4.45.2 tokenizers==0.20.3 && \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
        -r requirements.txt

COPY config.py retriever.py generator.py main.py server.py download_models.py ./

COPY models/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct /root/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct
COPY models/models--Dongjin-kr--ko-reranker /root/.cache/huggingface/hub/models--Dongjin-kr--ko-reranker
COPY models/models--bert-base-multilingual-cased /root/.cache/huggingface/hub/models--bert-base-multilingual-cased
COPY models/models--BAAI--bge-m3 /root/.cache/huggingface/hub/models--BAAI--bge-m3

COPY fastembed_cache/ /tmp/fastembed_cache/

COPY cache/ /app/cache/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV FASTEMBED_CACHE_PATH=/tmp/fastembed_cache/

EXPOSE 8000

CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "35"]
