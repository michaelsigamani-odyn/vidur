import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from vidur.config.device_sku_config import BaseDeviceSKUConfig


@dataclass
class ValidationIssue:
    level: str
    check: str
    file_path: str
    message: str


@dataclass
class ValidationSummary:
    hard_failures: int
    warnings: int


TDP_WATTS: Dict[str, float] = {
    "mi300x": 750.0,
    "radeon_8060s": 120.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate profiling CSVs for AMD SKUs.")
    parser.add_argument("--compute-root", type=str, default="data/profiling/compute")
    parser.add_argument("--device-skus", nargs="+", default=["mi300x", "radeon_8060s"])
    parser.add_argument("--repeat-root", type=str, default="")
    return parser.parse_args()


def _timing_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column.startswith("time_stats.")]


def _median_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in _timing_columns(df) if column.endswith(".median")]


def _read_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def _issue(level: str, check: str, file_path: Path, message: str) -> ValidationIssue:
    return ValidationIssue(level, check, str(file_path), message)


def _estimate_mlp_flops(row: pd.Series) -> float:
    tokens = float(row["num_tokens"])
    embd = float(row["n_embd"])
    hidden = float(row["n_expanded_embd"])
    gated_multiplier = 6.0 if bool(row["use_gated_mlp"]) else 4.0
    return tokens * embd * hidden * gated_multiplier


def _estimate_mlp_bytes(row: pd.Series) -> float:
    embd = float(row["n_embd"])
    hidden = float(row["n_expanded_embd"])
    gated_multiplier = 3.0 if bool(row["use_gated_mlp"]) else 2.0
    return embd * hidden * gated_multiplier * 2.0


def _measured_seconds(row: pd.Series, median_columns: List[str]) -> float:
    values = row[median_columns].to_numpy(dtype=float)
    return float(np.nan_to_num(values, nan=0.0).sum()) / 1000.0


def validate_completeness(file_path: Path) -> List[ValidationIssue]:
    dataframe = _read_csv(file_path)
    timing_columns = _timing_columns(dataframe)
    null_count = int(dataframe[timing_columns].isnull().sum().sum()) if timing_columns else 0
    if null_count == 0:
        return []
    return [_issue("hard", "completeness", file_path, f"null timing values: {null_count}")]


def validate_unsupported_points(model_dir: Path) -> List[ValidationIssue]:
    unsupported = model_dir / "unsupported_points.csv"
    if not unsupported.exists():
        return []
    dataframe = _read_csv(unsupported)
    if "reason" not in dataframe.columns:
        return [_issue("hard", "unsupported_points", unsupported, "missing reason column")]
    missing = int(dataframe["reason"].isnull().sum())
    if missing == 0:
        return []
    return [_issue("hard", "unsupported_points", unsupported, f"rows without reason: {missing}")]


def validate_roofline(file_path: Path, device_sku: str) -> List[ValidationIssue]:
    dataframe = _read_csv(file_path)
    config = BaseDeviceSKUConfig.create_from_type_string(device_sku)
    medians = _median_columns(dataframe)
    issues: List[ValidationIssue] = []
    for index, row in dataframe.iterrows():
        lower = _estimate_mlp_flops(row) / (config.fp16_tflops * 1e12)
        lower = max(lower, _estimate_mlp_bytes(row) / (config.memory_bandwidth_gbps * 1e9))
        if _measured_seconds(row, medians) >= lower:
            continue
        issues.append(
            _issue(
                "hard",
                "roofline_lower_bound",
                file_path,
                f"row={index} measured below roofline: measured_s={_measured_seconds(row, medians):.6e}, lower_s={lower:.6e}",
            )
        )
    return issues


def _group_median(dataframe: pd.DataFrame, key: List[str], value: str) -> pd.DataFrame:
    return dataframe.groupby(key, as_index=False)[value].median()


def _monotonicity_violations(dataframe: pd.DataFrame, x_name: str, y_name: str, key: List[str]) -> int:
    violation_count = 0
    for _, group in dataframe.groupby(key):
        ordered = group.sort_values(x_name)
        diffs = ordered[y_name].diff().fillna(0)
        violation_count += int((diffs < 0).sum())
    return violation_count


def validate_mlp_monotonicity(file_path: Path) -> List[ValidationIssue]:
    dataframe = _read_csv(file_path)
    medians = _median_columns(dataframe)
    dataframe["total_time_median"] = dataframe[medians].fillna(0).sum(axis=1)
    grouped = _group_median(
        dataframe,
        ["n_head", "n_kv_head", "n_embd", "n_expanded_embd", "vocab_size", "use_gated_mlp", "num_tensor_parallel_workers", "num_tokens"],
        "total_time_median",
    )
    violations = _monotonicity_violations(
        grouped,
        "num_tokens",
        "total_time_median",
        ["n_head", "n_kv_head", "n_embd", "n_expanded_embd", "vocab_size", "use_gated_mlp", "num_tensor_parallel_workers"],
    )
    if violations == 0:
        return []
    return [_issue("warn", "mlp_monotonicity", file_path, f"decreasing token-time pairs: {violations}")]


def validate_attention_monotonicity(file_path: Path) -> List[ValidationIssue]:
    dataframe = _read_csv(file_path)
    prefill = "time_stats.attn_prefill.median"
    decode = "time_stats.attn_decode.median"
    dataframe["attention_time_median"] = dataframe[[prefill, decode]].fillna(0).sum(axis=1)
    grouped = _group_median(
        dataframe,
        ["n_embd", "n_q_head", "n_kv_head", "num_tensor_parallel_workers", "batch_size", "prefill_chunk_size", "is_prefill", "kv_cache_size"],
        "attention_time_median",
    )
    violations = _monotonicity_violations(
        grouped,
        "kv_cache_size",
        "attention_time_median",
        ["n_embd", "n_q_head", "n_kv_head", "num_tensor_parallel_workers", "batch_size", "prefill_chunk_size", "is_prefill"],
    )
    if violations == 0:
        return []
    return [_issue("warn", "attention_monotonicity", file_path, f"decreasing kv-time pairs: {violations}")]


def validate_repeat_variance(file_a: Path, file_b: Path, join_columns: List[str]) -> List[ValidationIssue]:
    left = _read_csv(file_a)
    right = _read_csv(file_b)
    left["_value"] = left[_median_columns(left)].fillna(0).sum(axis=1)
    right["_value"] = right[_median_columns(right)].fillna(0).sum(axis=1)
    merged = left.merge(right, on=join_columns, suffixes=("_a", "_b"))
    if merged.empty:
        return [_issue("hard", "repeat_variance", file_a, "no joinable rows across repeats")]
    rel = (merged["_value_a"] - merged["_value_b"]).abs() / merged[["_value_a", "_value_b"]].mean(axis=1)
    p50 = float(np.nanmedian(rel))
    p95 = float(np.nanpercentile(rel, 95))
    if p50 < 0.03 and p95 < 0.10:
        return []
    return [_issue("hard", "repeat_variance", file_a, f"unstable repeats: median={p50:.4f}, p95={p95:.4f}")]


def validate_telemetry(model_dir: Path, device_sku: str) -> List[ValidationIssue]:
    telemetry_path = model_dir / "gpu_telemetry.jsonl"
    if not telemetry_path.exists():
        return [_issue("warn", "telemetry", telemetry_path, "missing gpu_telemetry.jsonl")]
    records = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    powers = [record.get("power_draw_watts") for record in records if record.get("power_draw_watts") is not None]
    if not powers:
        return [_issue("warn", "telemetry", telemetry_path, "power samples missing")]
    tdp = TDP_WATTS.get(device_sku)
    if tdp is None:
        return [_issue("warn", "telemetry", telemetry_path, f"no TDP baseline for {device_sku}")]
    mean_power = float(np.mean(powers))
    if 0.4 * tdp <= mean_power <= 1.0 * tdp:
        return []
    return [_issue("hard", "telemetry", telemetry_path, f"mean power {mean_power:.2f}W out of expected [40%,100%] TDP")]


def _compute_model_dirs(compute_root: Path, device_sku: str) -> List[Path]:
    return [path for path in (compute_root / device_sku).glob("*/*") if path.is_dir()]


def _repeat_file(repeat_root: Path, file_path: Path, compute_root: Path) -> Path:
    return repeat_root / file_path.relative_to(compute_root)


def _append(issues: List[ValidationIssue], new_issues: Iterable[ValidationIssue]) -> None:
    issues.extend(list(new_issues))


def validate_device(compute_root: Path, repeat_root: Path, device_sku: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for model_dir in _compute_model_dirs(compute_root, device_sku):
        mlp = model_dir / "mlp.csv"
        attention = model_dir / "attention.csv"
        if mlp.exists():
            _append(issues, validate_completeness(mlp))
            _append(issues, validate_roofline(mlp, device_sku))
            _append(issues, validate_mlp_monotonicity(mlp))
            if repeat_root.exists() and _repeat_file(repeat_root, mlp, compute_root).exists():
                _append(
                    issues,
                    validate_repeat_variance(
                        mlp,
                        _repeat_file(repeat_root, mlp, compute_root),
                        ["n_head", "n_kv_head", "n_embd", "n_expanded_embd", "vocab_size", "use_gated_mlp", "num_tokens", "num_tensor_parallel_workers"],
                    ),
                )
        if attention.exists():
            _append(issues, validate_completeness(attention))
            _append(issues, validate_attention_monotonicity(attention))
            if repeat_root.exists() and _repeat_file(repeat_root, attention, compute_root).exists():
                _append(
                    issues,
                    validate_repeat_variance(
                        attention,
                        _repeat_file(repeat_root, attention, compute_root),
                        ["n_embd", "n_q_head", "n_kv_head", "block_size", "num_tensor_parallel_workers", "max_model_len", "batch_size", "prefill_chunk_size", "kv_cache_size", "is_prefill", "attention_backend"],
                    ),
                )
        _append(issues, validate_unsupported_points(model_dir))
        _append(issues, validate_telemetry(model_dir, device_sku))
    return issues


def summarize(issues: List[ValidationIssue]) -> ValidationSummary:
    hard_failures = len([issue for issue in issues if issue.level == "hard"])
    warnings = len([issue for issue in issues if issue.level == "warn"])
    return ValidationSummary(hard_failures, warnings)


def print_issues(issues: List[ValidationIssue]) -> None:
    for issue in issues:
        print(f"[{issue.level}] {issue.check} {issue.file_path}: {issue.message}")


def main() -> int:
    args = parse_args()
    compute_root = Path(args.compute_root)
    repeat_root = Path(args.repeat_root) if args.repeat_root else Path("__missing_repeat_root__")
    issues = [
        issue
        for sku in args.device_skus
        for issue in validate_device(compute_root, repeat_root, sku)
    ]
    summary = summarize(issues)
    print_issues(issues)
    print(f"Validation summary: hard_failures={summary.hard_failures}, warnings={summary.warnings}")
    return 1 if summary.hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
