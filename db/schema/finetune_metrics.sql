BEGIN;

CREATE TABLE IF NOT EXISTS training_cluster (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    fabric TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_node (
    id BIGSERIAL PRIMARY KEY,
    cluster_id BIGINT NOT NULL REFERENCES training_cluster(id) ON DELETE CASCADE,
    hostname TEXT NOT NULL,
    mgmt_ip INET,
    fabric_ip INET,
    gpu_name TEXT,
    gpu_count INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cluster_id, hostname)
);

CREATE TABLE IF NOT EXISTS finetune_run (
    id BIGSERIAL PRIMARY KEY,
    run_key TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    precision TEXT,
    world_size INT,
    max_steps INT,
    learning_rate DOUBLE PRECISION,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS finetune_run_node (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES finetune_run(id) ON DELETE CASCADE,
    node_id BIGINT NOT NULL REFERENCES training_node(id) ON DELETE CASCADE,
    node_rank INT NOT NULL,
    local_rank INT,
    log_path TEXT,
    UNIQUE (run_id, node_rank)
);

CREATE TABLE IF NOT EXISTS finetune_step_metric (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES finetune_run(id) ON DELETE CASCADE,
    step INT NOT NULL,
    epoch DOUBLE PRECISION,
    loss DOUBLE PRECISION,
    grad_norm DOUBLE PRECISION,
    learning_rate DOUBLE PRECISION,
    samples_per_second DOUBLE PRECISION,
    steps_per_second DOUBLE PRECISION,
    runtime_seconds DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, step)
);

CREATE TABLE IF NOT EXISTS finetune_artifact (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES finetune_run(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    checkpoint_step INT,
    sha256 TEXT,
    size_bytes BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_finetune_run_model ON finetune_run(model_name);
CREATE INDEX IF NOT EXISTS idx_finetune_run_status ON finetune_run(status);
CREATE INDEX IF NOT EXISTS idx_step_metric_run ON finetune_step_metric(run_id, step);
CREATE INDEX IF NOT EXISTS idx_artifact_run ON finetune_artifact(run_id);

COMMIT;
