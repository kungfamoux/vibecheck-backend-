# Use Python 3.9.18 explicitly
FROM python:3.9.18-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip==22.3.1 \
    && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Download NLTK data
RUN python -m nltk.downloader punkt

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "4"]
