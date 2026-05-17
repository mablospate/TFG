-- Supabase schema for TFG quantum benchmark results.
-- Run once in Supabase SQL Editor: Dashboard → SQL Editor → New query.

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                          BIGSERIAL PRIMARY KEY,

    -- Run identity (same for all rows of a single benchmark run)
    run_id                      TEXT        NOT NULL,
    contributor                 TEXT,
    platform_id                 TEXT,
    benchmark_image             TEXT,
    gpu_enabled                 BOOLEAN,
    emulated                    BOOLEAN,

    -- Hardware (same for all rows of a single benchmark run)
    cpu_model                   TEXT,
    cpu_physical_cores          INT,
    cpu_logical_cores           INT,
    cpu_freq_mhz                REAL,
    ram_total_gb                REAL,
    gpu_model                   TEXT,
    gpu_vram_gb                 REAL,
    os                          TEXT,
    os_version                  TEXT,

    -- Benchmark identity
    algorithm                   TEXT        NOT NULL,  -- 'grover' | 'shor'
    framework                   TEXT        NOT NULL,
    framework_version           TEXT,
    n_qubits                    INT,
    num_shots                   INT,
    n_repetitions               INT,
    runtime_version             TEXT,
    python_version              TEXT,

    -- Per-repetition data
    repetition_index            INT         NOT NULL,  -- 0-based
    wall_time_ms                REAL,                  -- wall clock time for this repetition
    timestamp                   TIMESTAMPTZ,

    -- Aggregated stats (same for all reps of this framework-n combo)
    wall_time_median_ms         REAL,
    wall_time_iqr_ms            REAL,
    wall_time_mean_ms           REAL,
    wall_time_std_ms            REAL,
    build_time_ms               REAL,
    simulation_time_ms          REAL,
    startup_time_ms             REAL,
    subprocess_wall_time_ms     REAL,
    peak_memory_rss_mb          REAL,
    cpu_percent_mean            REAL,
    jsd                         REAL,
    cv                          REAL,

    -- Status
    status                      TEXT        NOT NULL DEFAULT 'ok',  -- 'ok' | 'error'
    error                       TEXT,

    -- Scaling curves (backfilled via PATCH after all n values complete)
    scaling_alpha               REAL,
    scaling_beta                REAL,
    scaling_data                JSONB,

    -- Shor-specific (NULL for Grover rows)
    n_to_factor                 INT,
    factor_found                INT,
    success_rate                REAL,

    -- QDisLib circuit cutting (NULL for non-qdislib rows)
    cutting_wall_time_ms        REAL,
    cutting_find_time_ms        REAL,
    cutting_expectation_value   REAL
);

CREATE INDEX IF NOT EXISTS idx_br_run_id  ON benchmark_runs (run_id);
CREATE INDEX IF NOT EXISTS idx_br_fw_algo ON benchmark_runs (framework, algorithm);
CREATE INDEX IF NOT EXISTS idx_br_contrib ON benchmark_runs (contributor);
