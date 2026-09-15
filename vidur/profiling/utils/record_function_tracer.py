import json
import os
import time
import uuid

import numpy as np
import torch


class RecordFunctionTracer:
    def __init__(self, output_path: str):
        trace_id = str(uuid.uuid4())[:8]
        self.trace_path = (
            f"{output_path}/profiler_traces/profiler_trace_{trace_id}.json"
        )
        self.enter_wall_ms = 0.0
        self.exit_wall_ms = 0.0
        self._events = None

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        self.enter_wall_ms = (time.perf_counter() - enter_start) * 1e3

    def __exit__(self, *args):
        exit_start = time.perf_counter()
        self.profiler.__exit__(None, None, None)
        torch.cuda.synchronize()
        self.profiler.export_chrome_trace(self.trace_path)

    def find_children(self, trace, event):
        if not ("dur" in event and "ts" in event):
            return

        children = []
        for e in trace:
            if not ("dur" in e and "ts" in e):
                continue

            # if the ts of the child is completely within the ts of the parent
            if (
                e["ts"] > event["ts"]
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]
            ):
                children.append(e)
        return children

    def find_correlated_event(self, trace, event):
        if not ("args" in event and "correlation" in event["args"]):
            return

        for e in trace:
            if not ("args" in e and "correlation" in e["args"]):
                continue

            if e == event:
                continue

            if e["args"]["correlation"] == event["args"]["correlation"]:
                return e

    def get_operation_time_stats(self):
        stats = {}

        if self._events is not None:
            trace = [event.key_averages_entry.trace_name for event in []]
            del trace
            trace = []
            for event in self._events:
                trace.append(
                    {
                        "cat": event.trace_name.split("::")[0]
                        if "::" in event.trace_name
                        else event.trace_name,
                        "name": event.name,
                        "ts": event.time_range.start,
                        "dur": event.time_range.elapsed_us(),
                    }
                )
            trace.extend(
                json.load(open(self.trace_path, "r"))["traceEvents"]
                if not trace and os.path.exists(self.trace_path)
                else []
            )
        else:
            trace = json.load(open(self.trace_path, "r"))["traceEvents"]

        for event in trace:
            if not ("cat" in event and event["cat"] == "user_annotation"):
                continue
            children = self.find_children(trace, event)
            cuda_time = 0
            for child in children:
                if not ("cat" in child and child["cat"] == "cuda_runtime"):
                    continue
                correlated_event = self.find_correlated_event(trace, child)
                if not correlated_event:
                    continue
                cuda_time += correlated_event["dur"]
            if cuda_time == 0:
                continue

            name = event["name"].replace("vidur_", "")

            if name not in stats:
                stats[name] = []

            stats[name].append(cuda_time * 1e-3)  # to convert to ms

        return {
            operation: {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }
            for operation, times in stats.items()
        }
