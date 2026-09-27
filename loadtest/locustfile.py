"""ITCS355 Lab 3 — load test (Locust alternative to k6).

    locust -f loadtest/locustfile.py --host https://<endpoint> \
           --users 10 --spawn-rate 5 --run-time 60s --headless

Use either this or k6, not both. Whichever you choose, commit it.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, constant, task

PAYLOAD_BYTES = max(0, int(os.environ.get("PAYLOAD_BYTES", "0")))


def sample_payload() -> dict:
    payload = {
        "temp_c": round(random.uniform(60, 95), 3),
        "vibration_mm_s": round(random.uniform(1.0, 8.0), 3),
        "pressure_kpa": round(random.uniform(280, 350), 3),
        "hours_since_service": round(random.uniform(0, 9000), 3),
        "load_pct": round(random.uniform(20, 100), 3),
        "ambient_humidity": round(random.uniform(30, 85), 3),
    }
    if PAYLOAD_BYTES > 300:
        payload["metadata"] = {"padding": "x" * (PAYLOAD_BYTES - 300)}
    return payload


class PredictUser(HttpUser):
    wait_time = constant(0)

    @task(9)
    def predict(self):
        with self.client.post("/predict", json=sample_payload(), catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")

    @task(1)
    def predict_batch(self):
        rows = [sample_payload() for _ in range(50)]
        with self.client.post("/predict/batch", json={"rows": rows}, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
            elif len(r.json().get("probabilities", [])) != len(rows):
                r.failure("batch response length mismatch")
