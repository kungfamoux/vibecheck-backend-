#!/bin/bash

# Exit on error and print commands
set -ex

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    . venv/bin/activate
fi

# Install system dependencies if needed (for Debian/Ubuntu)
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        python3.9 \
        python3-pip \
        python3.9-dev \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
fi

# Ensure Python 3.9 is used
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Install Python dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip==22.3.1
pip install setuptools==65.5.1 wheel==0.40.0
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -m nltk.downloader punkt

# Run the application
echo "Starting server..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
