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

# Move files to correct locations (robust to missing file types and subdirs)
RUN cp -a "IP-2 SignSpeeks/." /app/ && \
    find /app -maxdepth 1 -name "*.html" -exec mv {} /app/templates/ \; && \
    find /app -maxdepth 1 -name "*.jpg" -exec mv {} /app/static/ \; && \
    find /app -maxdepth 1 -name "*.jpeg" -exec mv {} /app/static/ \; && \
    find /app -maxdepth 1 -name "*.png" -exec mv {} /app/static/ \; && \
    find /app -maxdepth 1 -name "*.ico" -exec mv {} /app/static/ \; && \
    if [ -f /app/model.p ]; then mv /app/model.p /app/static/; fi && \
    chmod -R 755 /app/static /app/templates

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Expose the port provided by Railway
EXPOSE ${PORT}

# Run the application with more workers and longer timeout, using the PORT env variable
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 4 --timeout 120 --access-logfile - --error-logfile - web_app:app 