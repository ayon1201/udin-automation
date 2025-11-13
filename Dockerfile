FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads downloads

# Don't use ENV PORT here, let Render inject it
EXPOSE 10000

# Use ENTRYPOINT instead of CMD (harder to override)
ENTRYPOINT ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "--access-logfile", "-", "--error-logfile", "-"]
