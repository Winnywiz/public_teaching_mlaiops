"""Lab 5 Task 4 — prove the portability seam is real.

    python scripts/portability_swap_check.py --second-provider gcp

Exercises upload, download, and invoke against a SECOND provider's adapter. You are not
migrating the whole system; you are proving the seam exists.

If this fails while `make portability-audit` passes, you have found something worth
writing about: the leak was in configuration or in an assumption, not in an import.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-provider", required=True, choices=["aws", "azure", "gcp"])
    ap.add_argument("--second-blob-uri", required=True,
                    help="second provider's bucket/container prefix, for example s3://bucket/lab5")
    ap.add_argument("--invoke-endpoint", required=True,
                    help="existing second-provider endpoint; this script does not create one")
    ap.add_argument("--payload", type=Path, required=True,
                    help="JSON request body matching the deployed endpoint contract")
    ap.add_argument("--region", help="second provider's region; defaults to configured REGION")
    args = ap.parse_args()

    primary = config.load(strict=False)
    if primary.provider == args.second_provider:
        print("The second provider must differ from CLOUD_PROVIDER.")
        return 1

    print(f"primary   {primary.provider}")
    print(f"secondary {args.second_provider}\n")

    # Same Config object, different adapter. If your code needs more than this to switch,
    # say so in the write-up — that IS the finding.
    if not args.payload.is_file():
        ap.error(f"payload file does not exist: {args.payload}")
    try:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        ap.error(f"payload must be valid JSON: {exc}")
    if not isinstance(payload, dict):
        ap.error("payload JSON must be an object")

    swapped = config.Config(**{
        **primary.__dict__,
        "provider": args.second_provider,
        "blob_uri": args.second_blob_uri,
        "region": args.region or primary.region,
    })
    adapter = get_adapter(swapped)

    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.txt"
        probe.write_text("itcs355 lab5 portability probe\n", encoding="utf-8")
        key = f"lab5-portability/{uuid.uuid4().hex}.txt"
        uploaded_uri = None
        try:
            uploaded_uri = adapter.upload(str(probe), key, lab=5)
            results["upload"] = f"PASS — {uploaded_uri}"
        except NotImplementedError:
            results["upload"] = "NOT IMPLEMENTED"
        except Exception as exc:
            results["upload"] = f"FAIL — {type(exc).__name__}: {exc}"

        if uploaded_uri:
            try:
                downloaded = Path(tmp) / "back.txt"
                adapter.download(uploaded_uri, str(downloaded))
                if downloaded.read_text(encoding="utf-8") != probe.read_text(encoding="utf-8"):
                    raise RuntimeError("downloaded bytes differ from uploaded probe")
                results["download"] = f"PASS — round-tripped {uploaded_uri}"
            except NotImplementedError:
                results["download"] = "NOT IMPLEMENTED"
            except Exception as exc:
                results["download"] = f"FAIL — {type(exc).__name__}: {exc}"
        else:
            results["download"] = "SKIPPED — upload did not succeed"

        try:
            result = adapter.invoke(args.invoke_endpoint, payload)
            if not isinstance(result, dict):
                raise RuntimeError("invoke did not return a JSON object")
            results["invoke"] = "PASS — JSON response received"
        except NotImplementedError:
            results["invoke"] = "NOT IMPLEMENTED"
        except Exception as exc:
            results["invoke"] = f"FAIL — {type(exc).__name__}: {exc}"

        cleanup = getattr(adapter, "delete_uploaded_object", None)
        if uploaded_uri and cleanup:
            try:
                cleanup(uploaded_uri)
                results["cleanup"] = "PASS — temporary object removed"
            except Exception as exc:
                results["cleanup"] = f"FAIL — {type(exc).__name__}: {exc}"
        elif uploaded_uri:
            results["cleanup"] = "NOT SUPPORTED — remove the tagged probe object manually"

    for name, outcome in results.items():
        print(f"  [{outcome.split(' —')[0]:<16}] {name}  {outcome.partition('— ')[2]}")

    print("\nWrite-up (Lab 5 Task 4), half a page:")
    print("  - which method was hardest to port, and why")
    print("  - one place the abstraction genuinely leaked and could not be hidden")
    print("  - what a full migration would cost in engineering days")
    print("  - was building this abstraction worth it? A well-argued 'no' scores full marks.")

    return 0 if all(v.startswith("PASS") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
