"""Run the committed Locust script as a repeatable concurrency or payload matrix."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def locust_interpreter() -> str:
    try:
        import locust  # noqa: F401
        return sys.executable
    except ImportError:
        candidates = (
            ROOT / ".venv-loadtest" / "Scripts" / "python.exe",
            ROOT / ".venv-loadtest" / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise SystemExit(
            "Locust is not installed. Create .venv-loadtest and install "
            "requirements-loadtest.txt, or install that lock into the active Python."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("concurrency", "payload"), default="concurrency")
    parser.add_argument("--target", default="http://127.0.0.1:8080")
    parser.add_argument("--duration", default="30s")
    parser.add_argument("--p95-target-ms", type=float, default=200)
    parser.add_argument("--instance", default="local")
    parser.add_argument("--output-prefix", default="outputs/lab3")
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    if args.scenario == "concurrency":
        matrix = [(users, 0) for users in (1, 10, 50)]
    else:
        matrix = [(10, size) for size in (0, 4096, 16384, 60000)]
    loadtest_python = locust_interpreter()

    for users, payload_bytes in matrix:
        prefix = ROOT / f"{args.output_prefix}-{('vus' + str(users)) if args.scenario == 'concurrency' else ('payload' + str(payload_bytes))}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PAYLOAD_BYTES"] = str(payload_bytes)
        command = [
            loadtest_python, "-m", "locust", "-f", str(ROOT / "loadtest" / "locustfile.py"),
            "--host", args.target,
            "--users", str(users), "--spawn-rate", str(users),
            "--run-time", args.duration, "--headless", "--only-summary",
            "--csv", str(prefix),
        ]
        print(f"Running {args.scenario}: users={users}, payload_bytes={payload_bytes}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode:
            return result.returncode

    if args.scenario == "concurrency":
        report = [
            sys.executable, str(ROOT / "scripts" / "write_load_report.py"),
            "--target", args.target,
            "--p95-target-ms", str(args.p95_target_ms),
            "--instance", args.instance,
            "--duration", args.duration,
            "--payload-bytes", "0",
            "--output-prefix", args.output_prefix,
        ]
    else:
        report = [
            sys.executable, str(ROOT / "scripts" / "write_payload_report.py"),
            "--target", args.target, "--instance", args.instance,
            "--output-prefix", args.output_prefix,
        ]
    if args.report_out:
        report.extend(["--out", str(args.report_out)])
    return subprocess.run(report, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
