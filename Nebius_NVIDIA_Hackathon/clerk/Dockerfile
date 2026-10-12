# ---- 1. build the React frontend --------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npx vite build

# ---- 2. Python runtime serving the API and the built frontend ---------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLERK_STATIC_DIR=/app/static \
    CLERK_DB_PATH=/data/clerk.db \
    PORT=8000
WORKDIR /app
COPY backend/pyproject.toml ./
COPY backend/clerk ./clerk
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 clerk && mkdir -p /data && chown clerk /data
COPY --from=web /web/dist ./static
USER clerk
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"
CMD ["sh", "-c", "uvicorn clerk.api:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
