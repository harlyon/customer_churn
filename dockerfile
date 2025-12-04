# Use a slim version of Python to keep the image size down
# 3.10 is a stable choice for modern ML libraries
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Ensures logs are flushed directly to terminal
# PIP_NO_CACHE_DIR: Prevents pip from caching packages
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/reports && \
    chmod -R 777 /app/models /app/reports

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
# --host 0.0.0.0 makes the server publicly available
# --reload enables auto-reload in development (remove in production)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]