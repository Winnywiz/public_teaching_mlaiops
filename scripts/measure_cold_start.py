"""Wait for zero replicas, then record time to the first successful prediction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.azure import AzureAdapter
from src import config

PAYLOAD = {
    "temp_c": 78.4,
    "vibration_mm_s": 3.1,
    "pressure_kpa": 315.2,
    "hours_since_service": 4200.0,
    "load_pct": 68.0,
    "ambient_humidity": 55.0,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="itcs355-lab3")
    parser.add_argument("--wait-timeout", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-cold-start.md"))
    args = parser.parse_args()

    cfg = config.load(strict=True)
    if cfg.provider != "azure":
        raise SystemExit("Cold-start measurement requires the Azure Container Apps revision adapter.")
    adapter = AzureAdapter(cfg)
    revisions = [item for item in adapter.list_revisions(args.endpoint)
                 if item.get("active") and item.get("name")]
    if not revisions:
        raise SystemExit("No active revision found; deploy and smoke-test a model first.")
    revision = max(revisions, key=lambda item: int(item.get("traffic_weight", 0)))
    revision_name = str(revision["name"])
    app_name = adapter._containerapp_name(args.endpoint)

    zero_seen = None
    deadline = time.monotonic() + args.wait_timeout
    stable_zero_polls = 0
    while time.monotonic() < deadline:
        replicas = adapter._run_az(
            "containerapp", "replica", "list",
            "--name", app_name,
            "--resource-group", adapter.resource_group,
            "--revision", revision_name,
            json_output=True,
        )
        if isinstance(replicas, list) and not replicas:
            stable_zero_polls += 1
            zero_seen = zero_seen or now()
            if stable_zero_polls >= 2:
                break
        else:
            stable_zero_polls = 0
            zero_seen = None
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError(
            f"Revision {revision_name} did not reach zero replicas within {args.wait_timeout}s; "
            "check minReplicas, active traffic, and the Azure console."
        )

    detail = adapter._containerapp_detail(args.endpoint)
    fqdn = detail.get("properties", {}).get("configuration", {}).get("ingress", {}).get("fqdn")
    if not fqdn:
        raise RuntimeError("Container App has no external ingress FQDN")
    url = f"https://{fqdn}/predict"
    started_utc = now()
    started = time.perf_counter()
    deadline = started + args.request_timeout
    attempts = 0
    final_response: dict | None = None
    last_status = None
    last_error = None
    while time.monotonic() < deadline:
        attempts += 1
        request = Request(
            url,
            data=json.dumps(PAYLOAD).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=min(30, args.request_timeout)) as response:
                last_status = response.status
                final_response = json.loads(response.read().decode("utf-8"))
            if last_status == 200:
                break
        except HTTPError as exc:
            last_status = exc.code
            last_error = exc.read().decode("utf-8", errors="replace")[:300]
            if last_status not in {502, 503, 504}:
                raise
        except URLError as exc:
            last_error = str(exc.reason)
        except TimeoutError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if last_status != 200 or not isinstance(final_response, dict):
        raise TimeoutError(
            f"No successful prediction after {args.request_timeout}s; last status={last_status}, "
            f"error={last_error}"
        )

    warm_started = time.perf_counter()
    with urlopen(Request(
        url,
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    ), timeout=30) as response:
        warm_status = response.status
        warm_response = json.loads(response.read().decode("utf-8"))
    warm_ms = (time.perf_counter() - warm_started) * 1000
    if warm_status != 200 or not warm_response.get("model_version"):
        raise RuntimeError("Warm follow-up did not return a successful versioned prediction")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Lab 3 — scale-to-zero cold-start evidence

- Revision observed at zero replicas: `{revision_name}`
- Zero replicas confirmed (UTC): {zero_seen}
- First request started (UTC): {started_utc}
- First successful request: HTTP {last_status} after **{elapsed_ms:.1f} ms** ({attempts} attempt(s))
- Immediately warm follow-up: HTTP {warm_status} in **{warm_ms:.1f} ms**
- Model version: `{final_response.get('model_version', 'missing')}`
- First response request ID: `{final_response.get('request_id', 'missing')}`
- Non-success response detail before readiness: `{last_error or 'none'}`

This includes scale-out and model load/readiness time; it is recorded separately from steady-state
load-test percentiles. The first request keeps the app warm and potentially billable; the assignment
requires tagged teardown after all evidence is captured.
""", encoding="utf-8")
    print(f"cold-start={elapsed_ms:.1f}ms warm={warm_ms:.1f}ms report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
