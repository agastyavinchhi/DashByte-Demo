FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends make procps && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m venv .venv && .venv/bin/pip install -r requirements.txt
COPY . .
ENV DASHBOARD_HOST=0.0.0.0 PYTHONUNBUFFERED=1
EXPOSE 8501
CMD make run && make logs
