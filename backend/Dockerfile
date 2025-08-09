FROM python:3.10-slim

WORKDIR /app

COPY api/requirements.txt ./api-requirements.txt
COPY backend/ml/requirements.txt ./ml-requirements.txt

RUN pip install --no-cache-dir -r api-requirements.txt && \
    pip install --no-cache-dir -r ml-requirements.txt

COPY api/ ./api/
COPY backend/ml/ ./backend/ml/

WORKDIR /app/api

# FastAPI port
EXPOSE 8000

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
