"""Initial schema: extensions, raw/features/app schemas, all Phase 1 tables.

Revision ID: 0001
Revises:
Create Date: 2026-05-02
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extensions
    # CREATE EXTENSION IF NOT EXISTS is idempotent; safe on fresh envs
    # even when the extension is already enabled via the dashboard.
    # ------------------------------------------------------------------
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis')
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ------------------------------------------------------------------
    # Schemas
    # ------------------------------------------------------------------
    op.execute('CREATE SCHEMA IF NOT EXISTS raw')
    op.execute('CREATE SCHEMA IF NOT EXISTS features')
    op.execute('CREATE SCHEMA IF NOT EXISTS app')

    # ==================================================================
    # raw schema
    # ==================================================================

    # ------------------------------------------------------------------
    # raw.rodent_inspections
    # Source: DOHMH rodent inspection records (Socrata p937-wjvj / jh4g-rp64).
    # Supervised label column: result = 'Active Rat Signs'.
    # Capped to last 3 years at ingest (T-05).
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.rodent_inspections (
            inspection_id       TEXT PRIMARY KEY,
            inspection_date     DATE NOT NULL,
            bbl                 TEXT,               -- 10-char zero-padded
            bin                 TEXT,
            borough             SMALLINT,
            block               INTEGER,
            lot                 INTEGER,
            result              TEXT NOT NULL,       -- 'Active Rat Signs' | 'Passed' | ...
            inspection_type     TEXT,
            job_progress        SMALLINT,
            geom                GEOMETRY(Point, 4326),
            ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute('CREATE INDEX ON raw.rodent_inspections USING GIST (geom)')
    op.execute('CREATE INDEX ON raw.rodent_inspections (bbl)')
    op.execute('CREATE INDEX ON raw.rodent_inspections (inspection_date)')

    # ------------------------------------------------------------------
    # raw.complaints_nta_week
    # Source: 311 rodent complaints (Socrata erm2-nwe9).
    # Per $0-budget amendment: aggregated to NTA-week at ingest — no
    # per-complaint rows are stored.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.complaints_nta_week (
            nta_id              TEXT NOT NULL,
            week_start          DATE NOT NULL,       -- Monday
            complaint_count     INTEGER NOT NULL DEFAULT 0,
            ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (nta_id, week_start)
        )
    """)
    op.execute('CREATE INDEX ON raw.complaints_nta_week (week_start)')

    # ------------------------------------------------------------------
    # raw.restaurant_inspections
    # Source: DOHMH restaurant inspections (Socrata 43nn-pn8j).
    # Pest-relevant violation codes: 04K (live roaches), 04L (evidence of
    # mice/rats), 08A (facility not free of pests).
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.restaurant_inspections (
            record_id           TEXT PRIMARY KEY,    -- camis||'_'||inspection_date||'_'||violation_code
            camis               TEXT NOT NULL,        -- unique restaurant identifier
            bbl                 TEXT,                -- 10-char zero-padded
            nta_id              TEXT,                -- populated by T-06 spatial join
            inspection_date     DATE NOT NULL,
            violation_code      TEXT,
            is_pest_violation   BOOLEAN NOT NULL DEFAULT FALSE,
            grade               TEXT,
            score               SMALLINT,
            ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute('CREATE INDEX ON raw.restaurant_inspections (bbl)')
    op.execute('CREATE INDEX ON raw.restaurant_inspections (nta_id, inspection_date)')

    # ------------------------------------------------------------------
    # raw.dob_permits
    # Source: DOB NOW Permits (Socrata ipu4-2q9a) +
    #         DOB Permit Issuance legacy (Socrata rbx6-tga4).
    # Incremental on issuance_date (T-07).
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.dob_permits (
            permit_key          TEXT PRIMARY KEY,    -- source-specific identifier
            bbl                 TEXT,               -- 10-char zero-padded
            bin                 TEXT,
            borough             TEXT,
            issuance_date       DATE,
            expiration_date     DATE,
            job_type            TEXT,               -- 'A1', 'A2', 'NB', 'DM', etc.
            work_type           TEXT,
            source              TEXT NOT NULL,      -- 'now' | 'legacy'
            ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute('CREATE INDEX ON raw.dob_permits (bbl)')
    op.execute('CREATE INDEX ON raw.dob_permits (issuance_date)')

    # ------------------------------------------------------------------
    # raw.weather_daily
    # Source: Meteostat station USW00094728 (Central Park).
    # Fields: tavg/tmin/tmax (°C), prcp (mm), snow (mm),
    #         hdd/cdd computed at ingest (base 18°C).
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.weather_daily (
            date                DATE PRIMARY KEY,
            tavg_c              NUMERIC,
            tmin_c              NUMERIC,
            tmax_c              NUMERIC,
            prcp_mm             NUMERIC,
            snow_mm             NUMERIC,
            hdd                 NUMERIC,            -- heating degree days, base 18°C
            cdd                 NUMERIC,            -- cooling degree days, base 18°C
            ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ------------------------------------------------------------------
    # raw.ingest_cursors
    # Tracks the last-seen cursor for each incremental ingest source so
    # scripts can resume without re-fetching historical data.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE raw.ingest_cursors (
            source              TEXT PRIMARY KEY,   -- e.g. '311_complaints', 'dob_permits'
            last_cursor_value   TEXT,               -- ISO date, offset, or opaque token
            last_run_at         TIMESTAMPTZ,
            rows_ingested_total BIGINT NOT NULL DEFAULT 0
        )
    """)

    # ==================================================================
    # features schema
    # ==================================================================

    # ------------------------------------------------------------------
    # features.nta_week_panel
    # Central feature table at NTA × week grain.
    # Clay PCA stored as VECTOR(32) per ADR-0002 instead of 32 separate
    # NUMERIC columns; use pgvector operators for distance queries.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE features.nta_week_panel (
            nta_id              TEXT NOT NULL,           -- NTA 2020 code
            week_start          DATE NOT NULL,           -- Monday

            -- Labels (populated by T-05 via T-08)
            active_rat_signs_count  INTEGER NOT NULL DEFAULT 0,
            inspections_count       INTEGER NOT NULL DEFAULT 0,
            active_rat_signs_rate   NUMERIC,             -- count / inspections
            active_rat_signs_ind    BOOLEAN,             -- count > 0

            -- 311 complaint features (lagged; populated by T-06 via T-08)
            complaints_count        INTEGER NOT NULL DEFAULT 0,
            complaints_lag_1w       INTEGER,
            complaints_lag_4w       INTEGER,
            complaints_lag_12w      INTEGER,

            -- Restaurant pest violations per NTA-week (T-07 via T-08)
            rest_pest_violations_count  INTEGER NOT NULL DEFAULT 0,

            -- DOB permit disturbance features (T-07 via T-08)
            permits_active_count    INTEGER NOT NULL DEFAULT 0,
            demolitions_count       INTEGER NOT NULL DEFAULT 0,

            -- Weather (joined on week_start Monday; T-07 via T-08)
            weather_tavg_c          NUMERIC,
            weather_prcp_mm         NUMERIC,
            weather_hdd             NUMERIC,
            weather_cdd             NUMERIC,

            -- PLUTO static aggregates per NTA (joined once; T-07 via T-08)
            units_total             INTEGER,
            year_built_median       INTEGER,
            landuse_residential_pct NUMERIC,
            landuse_commercial_pct  NUMERIC,

            -- Spatial lag features (populated by features/spatial_lags.py; T-09)
            neighbor_active_rat_signs_rate_lag_1w   NUMERIC,
            neighbor_complaints_count_lag_4w         NUMERIC,

            -- Regime indicator booleans (populated by features/regime_indicators.py; T-10)
            regime_covid                    BOOLEAN,    -- 2020-03-01 to 2020-06-30
            regime_8pm_setout               BOOLEAN,    -- >= 2023-04-01
            regime_commercial_containerization BOOLEAN, -- >= 2023-07-31 (full impl 2024-03-01)
            regime_residential_containerization BOOLEAN,-- >= 2024-11-01
            regime_rmz_active               BOOLEAN,    -- FALSE in Phase 1; updated later

            -- Clay 32-dim PCA embedding (ADR-0002: VECTOR(32) not 32 NUMERIC columns)
            clay_pca                        VECTOR(32),

            PRIMARY KEY (nta_id, week_start)
        )
    """)
    op.execute('CREATE INDEX ON features.nta_week_panel (week_start)')
    op.execute('CREATE INDEX ON features.nta_week_panel (nta_id)')

    # ==================================================================
    # app schema
    # ==================================================================

    # ------------------------------------------------------------------
    # app.health_code_chunks
    # RAG corpus: NYC Health Code and related legal documents.
    # Populated in Phase 4 (T-XX ingest_health_code scripts).
    # Created here so Phase 4 requires no schema migration.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE app.health_code_chunks (
            chunk_id            UUID PRIMARY KEY,
            document            TEXT NOT NULL,      -- 'hc_article_151', 'hmc_27_2017', etc.
            citation            TEXT NOT NULL,      -- '§151.02(a)(2)'
            authority           TEXT NOT NULL,      -- 'NYC DOHMH', 'NYC HPD', etc.
            section_path        TEXT[] NOT NULL,    -- ['151', '151.02', '151.02(a)']
            defined_terms       TEXT[],             -- terms extracted via regex
            cross_refs          TEXT[],             -- citation cross-references
            parent_chunk_id     UUID REFERENCES app.health_code_chunks,
            content             TEXT NOT NULL,
            content_with_prefix TEXT NOT NULL,      -- 'From NYC Health Code §151...: <content>'
            token_count         INTEGER NOT NULL,
            effective_date      DATE,
            version_hash        TEXT NOT NULL,
            embedding           VECTOR(1024),       -- voyage-3-large
            embedding_bge       VECTOR(1024),       -- BGE-M3 (ablation)
            content_tsv         TSVECTOR GENERATED ALWAYS AS (
                                    to_tsvector('english', content)
                                ) STORED
        )
    """)
    # HNSW index for dense retrieval (m=16, ef_construction=64 per spec §7.2)
    op.execute("""
        CREATE INDEX ON app.health_code_chunks
        USING HNSW (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute('CREATE INDEX ON app.health_code_chunks USING GIN (content_tsv)')
    op.execute('CREATE INDEX ON app.health_code_chunks USING GIN (content gin_trgm_ops)')

    # ------------------------------------------------------------------
    # app.chat_sessions
    # One row per conversation. user_fingerprint is an optional client-
    # side hash for analytics; no PII is stored.
    # RLS deferred — see ADR-0004.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE app.chat_sessions (
            session_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            user_fingerprint    TEXT
        )
    """)

    # ------------------------------------------------------------------
    # app.chat_messages
    # One row per turn. retrieved_chunks stores the chunk IDs + scores
    # used to generate the response (for eval and debug).
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE app.chat_messages (
            message_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            session_id          UUID NOT NULL REFERENCES app.chat_sessions,
            role                TEXT NOT NULL,      -- 'user' | 'assistant' | 'system'
            content             TEXT NOT NULL,
            retrieved_chunks    JSONB,              -- [{chunk_id, score}]
            trace_id            TEXT,               -- OpenInference trace ID
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            latency_ms          INTEGER,
            cost_usd            NUMERIC(10, 6)
        )
    """)
    op.execute('CREATE INDEX ON app.chat_messages (session_id, created_at)')

    # ------------------------------------------------------------------
    # app.risk_predictions
    # Cache of model output per NTA × week. top_factors is a JSONB array
    # of [{feature, contribution}] for the frontend attribution panel.
    # Created here (Phase 1) so Phase 2 can write without a migration.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE app.risk_predictions (
            nta_id              TEXT NOT NULL,
            predicted_for_week  DATE NOT NULL,
            risk_score          NUMERIC NOT NULL,   -- calibrated probability [0,1]
            risk_decile         SMALLINT NOT NULL,  -- 1–10
            top_factors         JSONB NOT NULL,     -- [{feature, contribution}]
            model_version       TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (nta_id, predicted_for_week, model_version)
        )
    """)
    op.execute('CREATE INDEX ON app.risk_predictions (predicted_for_week, risk_decile)')

    # ------------------------------------------------------------------
    # app.eval_runs
    # Stores nightly RAG eval results (Phase 4, T-XX).
    # Columns mirror the metrics reported in evals/results/<date>.json
    # and the Phase 4 acceptance check (Recall@5 >= 0.70).
    # Created here so Phase 4 requires no schema migration.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE app.eval_runs (
            run_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            run_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            eval_set            TEXT NOT NULL,      -- 'article151_qa_v1'
            model_id            TEXT NOT NULL,      -- 'claude-haiku-4-5'
            retriever_config    TEXT NOT NULL,      -- 'hnsw+bm25+rrf+bge_rerank'
            recall_at_5         NUMERIC,
            recall_at_10        NUMERIC,
            faithfulness        NUMERIC,
            citation_accuracy   NUMERIC,
            n_questions         INTEGER,
            metrics_json        JSONB NOT NULL DEFAULT '{}',
            git_sha             TEXT,
            notes               TEXT
        )
    """)
    op.execute('CREATE INDEX ON app.eval_runs (run_at DESC)')
    op.execute('CREATE INDEX ON app.eval_runs (eval_set, run_at DESC)')


def downgrade() -> None:
    # Drop in reverse dependency order.
    op.execute('DROP TABLE IF EXISTS app.eval_runs CASCADE')
    op.execute('DROP TABLE IF EXISTS app.risk_predictions CASCADE')
    op.execute('DROP TABLE IF EXISTS app.chat_messages CASCADE')
    op.execute('DROP TABLE IF EXISTS app.chat_sessions CASCADE')
    op.execute('DROP TABLE IF EXISTS app.health_code_chunks CASCADE')
    op.execute('DROP TABLE IF EXISTS features.nta_week_panel CASCADE')
    op.execute('DROP TABLE IF EXISTS raw.ingest_cursors CASCADE')
    op.execute('DROP TABLE IF EXISTS raw.weather_daily CASCADE')
    op.execute('DROP TABLE IF EXISTS raw.dob_permits CASCADE')
    op.execute('DROP TABLE IF EXISTS raw.restaurant_inspections CASCADE')
    op.execute('DROP TABLE IF EXISTS raw.complaints_nta_week CASCADE')
    op.execute('DROP TABLE IF EXISTS raw.rodent_inspections CASCADE')
    op.execute('DROP SCHEMA IF EXISTS app CASCADE')
    op.execute('DROP SCHEMA IF EXISTS features CASCADE')
    op.execute('DROP SCHEMA IF EXISTS raw CASCADE')
