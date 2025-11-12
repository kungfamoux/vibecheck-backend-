#!/bin/bash

# Exit on error
set -e

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -m nltk.downloader punkt

# Run the application
echo "Starting server..."
uvicorn main:app --host 0.0.0.0 --port $PORT
