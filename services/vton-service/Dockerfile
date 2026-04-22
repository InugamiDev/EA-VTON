FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ /app/src/

ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8002"]
