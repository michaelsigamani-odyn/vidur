import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class HostEnv:
    host_alias: str
    captured_at_utc: str
    command: str
    stdout: str
    stderr: str
    return_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture host environment evidence for profiling runs.")
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _capture_host(host_alias: str, remote_command: str) -> HostEnv:
    result = subprocess.run(["ssh", host_alias, remote_command], capture_output=True, text=True)
    now = datetime.now(timezone.utc).isoformat()
    return HostEnv(host_alias, now, remote_command, result.stdout, result.stderr, result.returncode)


def _remote_command() -> str:
    return "hostname; /opt/rocm/bin/amd-smi static 2>/dev/null | head -20 || amd-smi static 2>/dev/null | head -20 || nvidia-smi -L"


def capture(hosts: List[str]) -> List[HostEnv]:
    command = _remote_command()
    return [_capture_host(host, command) for host in hosts]


def _write_output(output_path: Path, records: List[HostEnv]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = capture(args.hosts)
    _write_output(Path(args.output), records)
    failed = [record for record in records if record.return_code != 0]
    print(json.dumps([asdict(record) for record in records], indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
