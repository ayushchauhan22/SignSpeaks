# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/static /app/templates

# Copy the application files
COPY . .

# Move files to correct locations
RUN cp -r "IP-2 SignSpeeks"/* /app/static/ && \
    cp -r "IP-2 SignSpeeks"/* /app/templates/ && \
    cp "IP-2 SignSpeeks/model.p" /app/static/ && \
    chmod -R 755 /app/static /app/templates

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV MODEL_PATH=/app/static/model.p
ENV FLASK_ENV=production

# Expose the port
EXPOSE 8000

# Run the application with more workers and longer timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "web_app:app"] 