#!/usr/bin/env bash
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr chromium chromium-driver

# Install Chrome dependencies
apt-get install -y libnss3 libgconf-2-4 libxi6 libxcursor1 libxss1 libxrandr2 libasound2 libpangocairo-1.0-0 libatk1.0-0 libcups2 libxcomposite1 libxdamage1
