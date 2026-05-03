from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    # Pooled URL (PgBouncer transaction mode) — used by the FastAPI app at runtime.
    database_url: str
    # Direct URL (bypasses PgBouncer) — used by Alembic for DDL migrations.
    direct_database_url: str

    # --- Supabase ---
    supabase_url: str
    supabase_anon_key: str

    # --- LLM / embeddings ---
    anthropic_api_key: str = ""
    voyageai_api_key: str = ""
    # Cohere is optional; enables Rerank 3.5 ablation when present.
    cohere_api_key: str = ""

    # --- Data sources ---
    nyc_socrata_app_token: str = ""
    census_api_key: str = ""  # ACS 5-year Table B19013 (T-03)

    # --- Observability ---
    phoenix_otlp_endpoint: str = "http://localhost:4317"
    otlp_json_sink: str = "logs/otlp_traces.jsonl"
    sentry_dsn: str = ""

    # --- Spend guard ---
    # Warn (log + Sentry alert) if cumulative daily LLM spend exceeds this.
    daily_spend_alert_usd: float = 1.0

    # Hardcoded price table (USD per 1M tokens) used by cost-logging middleware.
    # Update when Anthropic changes pricing.
    llm_price_per_1m_input: dict[str, float] = {
        "claude-haiku-4-5": 0.80,
        "claude-sonnet-4-5": 3.00,
    }
    llm_price_per_1m_output: dict[str, float] = {
        "claude-haiku-4-5": 4.00,
        "claude-sonnet-4-5": 15.00,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
