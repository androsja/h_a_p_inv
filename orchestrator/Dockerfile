FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir fastapi uvicorn[standard] aiofiles pydantic docker python-dotenv joblib scikit-learn numpy
COPY orchestrator/ ./orchestrator/
COPY shared/ ./shared/
RUN mkdir -p /app/data /app/logs
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
