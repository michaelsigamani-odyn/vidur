BEGIN;

INSERT INTO finetune_run (
    run_key,
    model_name,
    strategy,
    precision,
    world_size,
    max_steps,
    learning_rate,
    status,
    started_at,
    finished_at
) VALUES
    ('qwen3-tts-1p7b-dgx1-20260915', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', 'vllm-omni-serve', 'fp16', 1, NULL, NULL, 'failed_missing_deploy_config', '2026-09-15T04:12:54Z', '2026-09-15T04:15:39Z'),
    ('qwen3-tts-1p7b-dgx3-20260915', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', 'vllm-omni-serve', 'fp16', 1, NULL, NULL, 'failed_missing_deploy_config', '2026-09-15T04:31:29Z', '2026-09-15T04:34:14Z'),
    ('qwen25-mlp-mi300x-20260915', 'Qwen/Qwen2.5-7B-Instruct', 'vidur-mlp-profile', 'bf16', 1, NULL, NULL, 'blocked_tp_size_api_drift', '2026-09-15T03:35:00Z', NULL)
ON CONFLICT (run_key) DO UPDATE SET
    status = EXCLUDED.status,
    finished_at = EXCLUDED.finished_at,
    created_at = NOW();

COMMIT;
