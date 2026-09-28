FROM python:3.11-slim

WORKDIR /app

# system deps for pdfplumber + OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm -q

COPY . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "kgraph.server:app", "--host", "0.0.0.0", "--port", "8000"]
