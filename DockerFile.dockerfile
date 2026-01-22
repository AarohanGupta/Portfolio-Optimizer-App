# Base image: Lightweight Python 3.10
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first (for caching layers)
COPY requirements.txt .

# Install dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the container is running (good for cloud platforms)
HEALTHCHECK CMD curl --fail http://localhost:8501/_