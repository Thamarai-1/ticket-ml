FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run model training and retrieval index construction if artifacts missing
RUN python src/train.py || true
RUN python src/build_indexes.py || true

EXPOSE 8000

# Native Python Healthcheck (No extra curl dependency needed)
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode() == 200 else 1)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
