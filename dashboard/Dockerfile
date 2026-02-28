FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir fastapi uvicorn[standard] aiofiles pandas scikit-learn joblib python-dotenv yfinance
COPY dashboard/ ./dashboard/
COPY shared/ ./shared/
RUN mkdir -p /app/data /app/logs
CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
