"""Request and response contracts.

Validation lives here so a malformed request is rejected with 422 and a useful message,
rather than reaching the model and producing a confident number from nonsense.

The bounds mirror src/data.PLAUSIBLE_RANGES. Keep them in step: when Lab 4 adds a data
contract test, the same bounds are what CI asserts against.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    temp_c: float = Field(..., ge=-10, le=140, allow_inf_nan=False)
    vibration_mm_s: float = Field(..., ge=0, le=60, allow_inf_nan=False)
    pressure_kpa: float = Field(..., ge=0, le=600, allow_inf_nan=False)
    hours_since_service: float = Field(..., ge=0, le=20000, allow_inf_nan=False)
    load_pct: float = Field(..., ge=0, le=100, allow_inf_nan=False)
    ambient_humidity: float = Field(..., ge=0, le=100, allow_inf_nan=False)
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional caller metadata; ignored by the model and capped at 64 KiB.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("metadata")
    @classmethod
    def metadata_is_bounded(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is not None and len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 65_536:
            raise ValueError("metadata must be at most 65536 UTF-8 bytes")
        return value


class PredictResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    model_version: str


class BatchRequest(BaseModel):
    rows: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    probabilities: list[float]
    model_version: str
