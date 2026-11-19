FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    LMP_MODEL_PATH=/models/model.json \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r requirements-service.txt
COPY lmp.py service.py ./
RUN python -c "from lmp import MODEL, MODEL_REVISION; from sentence_transformers import SentenceTransformer; SentenceTransformer(MODEL, revision=MODEL_REVISION, device='cpu')" && \
    chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=2).close()"
CMD ["python", "-m", "uvicorn", "service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "32"]
