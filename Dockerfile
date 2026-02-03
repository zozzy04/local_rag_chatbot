FROM python:3.10-slim

WORKDIR /app

# Installazione dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installazione librerie Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamo il codice (anche se i volumi lo sovrascriveranno in dev)
COPY . .

EXPOSE 8000