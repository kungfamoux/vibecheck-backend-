#!/bin/bash

# Exit on error and print commands
set -ex

# Install system dependencies if needed
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        python3.9 \
        python3-pip \
        python3.9-dev \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
fi

# Create a virtual environment
echo "Setting up Python environment..."
python3.9 -m venv /opt/render/venv
source /opt/render/venv/bin/activate

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -m nltk.downloader punkt

# Run the application
echo "Starting server..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT
