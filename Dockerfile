# CPU image: FastAPI + chat UI. The corpus, index and Hugging Face models are created on first
# start (docker/entrypoint.sh) inside the /app/data and /app/models volumes, never baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf

WORKDIR /app

# CPU-only PyTorch first, so pip does not pull the multi-GB CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY eval ./eval
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/models \
    && chown -R app:app /app
USER app

VOLUME ["/app/data", "/app/models"]
EXPOSE 8000
# Pass the key at run time (-e ANTHROPIC_API_KEY or --env-file .env); without it the app
# still serves retrieval results.
ENTRYPOINT ["entrypoint.sh"]
CMD ["rag-serve", "--host", "0.0.0.0", "--port", "8000"]
