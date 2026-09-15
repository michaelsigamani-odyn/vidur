## Qwen2.5 QSFP fine-tuning snapshot

- Cluster: `odyn-dgx1` + `odyn-dgx3`
- Fabric: QSFP direct network (`192.168.100.10` <-> `192.168.100.11`)
- Runtime: `torchrun --nnodes 2 --nproc_per_node 1`
- Transport override: `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`
- Run root: `/home/michael/lf_runs/qwen_qsfp_20260915_0250`

### Captured artifacts

- Node logs in `logs/` for `qwen25_7b` and `qwen25_14b`
- Completed 7B trainer metrics: `metrics/qwen25_7b/checkpoint-120/trainer_state.json`
- 14B trainer metrics snapshots: `metrics/qwen25_14b/checkpoint-60/trainer_state.json`, `metrics/qwen25_14b/checkpoint-120/trainer_state.json`

### Notes

- 7B run completed and rolled into 14B in the same launcher sequence.
- 14B completed to step 120; both midpoint and final metric snapshots are stored.
