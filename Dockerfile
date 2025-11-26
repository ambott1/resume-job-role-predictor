FROM python:3.10-slim

# System dependencies (XGBoost + build tools for compiling C extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Flask app listens on 5000 (as in app.py)
EXPOSE 5000

# Train from scratch (saves to artifacts/) then start the web app
CMD ["sh", "-c", "python3 src/pipeline/train_pipeline.py && python3 app.py"]
