FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends make procps && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m venv .venv && .venv/bin/pip install -r requirements.txt
COPY . .
ENV DASHBOARD_HOST=0.0.0.0 PYTHONUNBUFFERED=1
EXPOSE 8501
# Start the stack in the background and stream its logs. On docker stop / Ctrl+C,
# run make stop so each stage shuts down cleanly instead of being killed.
CMD ["sh", "-c", "trap 'make stop; exit 0' TERM INT; make run && make logs & wait"]
