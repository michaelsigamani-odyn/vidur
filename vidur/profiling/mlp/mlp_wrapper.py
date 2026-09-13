import os
import time

import torch

from vidur.profiling.common.accelerator import get_torch_device, synchronize_device
from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.model_executor_backend import create_model_executor_backend

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.mlp.mlp_impl import GPTModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 20


class MlpWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        profile_method: str,
        rank: int,
        output_dir: str,
        gpu_vendor: str,
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.model_config = model_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.profile_method = profile_method
        self.rank = rank
        self.output_dir = output_dir
        self.device = get_torch_device()
        self.backend = create_model_executor_backend(gpu_vendor)
        self.backend.patch_cuda_timer(CudaTimer)
        self.backend.prepare_mlp_runtime(
            model_name=model_config.name,
            tensor_parallel_size=num_tensor_parallel_workers,
            rank=rank,
        )
        self._is_first_profile_call = True
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        with self.backend.mlp_execution_context():
            self.model = GPTModel(
                model_config,
                num_tensor_parallel_workers,
                self.backend,
                (
                    ACTIVE_STEPS
                    if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                    else 1
                ),
            )
            self.backend.initialize_dummy_weights(self.model)
            self.model = self.model.to(device=self.device, dtype=torch.float16).eval()

    def __del__(self):
        if hasattr(self, "backend"):
            self.backend.cleanup_mlp_runtime()

    @torch.inference_mode()
    def profile(self, num_tokens: int):
        with self.backend.mlp_execution_context():
            vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers
            input_ids = torch.randint(
                low=0,
                high=vocab_range,
                size=(num_tokens,),
                device=self.device,
                dtype=torch.long,
            )
            positions = torch.arange(num_tokens, device=self.device, dtype=torch.long)

            if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
                # Run the model once without capturing the graph.
                # This is to make sure that the captured graph does not include the
                # kernel launches for initial benchmarking (e.g., Triton autotune).
                warmup_start = time.perf_counter()
                self.model(
                    input_ids,
                    positions,
                )
                synchronize_device()
                warmup_wall_ms = (time.perf_counter() - warmup_start) * 1e3

                self.timer_stats_store.clear_stats()

                record_function_tracer = RecordFunctionTracer(self.output_dir)

                active_start = time.perf_counter()
                with record_function_tracer:
                    self.model(
                        input_ids,
                        positions,
                    )
                synchronize_device()
                active_wall_ms = (time.perf_counter() - active_start) * 1e3

                time_stats = record_function_tracer.get_operation_time_stats()
            else:
                warmup_start = time.perf_counter()
                for _ in range(WARMUP_STEPS):
                    self.model(
                        input_ids,
                        positions,
                    )

                synchronize_device()
                warmup_wall_ms = (time.perf_counter() - warmup_start) * 1e3

                self.timer_stats_store.clear_stats()

                active_start = time.perf_counter()
                for _ in range(ACTIVE_STEPS):
                    self.model(
                        input_ids,
                        positions,
                    )

                synchronize_device()
                active_wall_ms = (time.perf_counter() - active_start) * 1e3

                time_stats = self.timer_stats_store.get_stats()

            compile_warmup_wall_ms = warmup_wall_ms if self._is_first_profile_call else 0.0
            self._is_first_profile_call = False
            active_steps = 1 if self.profile_method == ProfileMethod.RECORD_FUNCTION.value else ACTIVE_STEPS
            measured_kernel_wall_ms = active_wall_ms
            measured_kernel_mean_ms = measured_kernel_wall_ms / max(active_steps, 1)

        stats = {
            "time_stats": time_stats,
            "compile_warmup_wall_ms": compile_warmup_wall_ms,
            "warmup_wall_ms": warmup_wall_ms,
            "measured_kernel_wall_ms": measured_kernel_wall_ms,
            "measured_kernel_mean_ms": measured_kernel_mean_ms,
            "active_steps": active_steps,
            "n_head": self.model_config.num_q_heads,
            "n_kv_head": self.model_config.num_kv_heads,
            "n_embd": self.model_config.embedding_dim,
            "n_expanded_embd": self.model_config.mlp_hidden_dim,
            "vocab_size": self.model_config.vocab_size,
            "use_gated_mlp": self.model_config.use_gated_mlp,
            "num_tokens": num_tokens,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
        self.timer_stats_store.clear_stats()

        return stats
