FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.joblib .
COPY synthetic_data.csv .

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]