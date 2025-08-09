FROM python:3.10-slim

WORKDIR /app

COPY ./frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./frontend/app.py .

# streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
