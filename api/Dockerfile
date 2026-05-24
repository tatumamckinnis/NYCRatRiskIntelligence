# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.4.29

# Copy and install dependencies first (layer cache)
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.6" \
        "pydantic-settings>=2.3" \
        "asyncpg>=0.29" \
        "numpy" \
        "pandas" \
        "scikit-learn" \
        "joblib" \
        "sentry-sdk[fastapi]>=2.13" \
        "python-dotenv>=1.0"

# Copy application source
COPY src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "rat_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
