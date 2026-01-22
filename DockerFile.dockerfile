# Base image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# --- NEW STEP: Upgrade pip and install 'wheel' first ---
# This forces the system to look for pre-built binaries instead of compiling from scratch
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies (now much faster)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
