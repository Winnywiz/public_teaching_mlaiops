"""Token accounting and the LLM cost model. Session 5, Lab 5 Part B.

An LLM endpoint is priced per token, not per hour, which breaks every intuition built in
Labs 2 and 3. A classifier that costs the same whether it answers 10 or 10,000 requests
becomes an assistant whose bill is a linear function of how verbose it is allowed to be.
`src/costs.py` prices provisioned compute; this module prices tokens. Lab 5 makes you
report both, because a real system has both.

Three levers move an LLM bill, in the order they are worth pulling:

  1. Output length.   Output tokens can cost several times more than input tokens.
                      Verify the exact model's rate, then evaluate an output cap.
  2. Prompt caching.  Cached input rates and cache-write premiums vary by provider.
                      Supply the exact factors; measure actual cached-token usage.
  3. Model choice.    A smaller model is often 10-20x cheaper. Whether it is good enough
                      is an evaluation question, not a pricing one — which is why
                      scripts/llm_eval.py exists and why you run it before you switch.

Prices are THB per 1,000,000 tokens. They change often and differ by model, region, and
service tier. This module intentionally ships with no provider price table: add the
rates copied from the actual provider pricing page when running the report. Synthetic
offline fixtures must never be presented as actual provider usage.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Usage:
    """One request's token usage, as every provider reports it under different names."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                "cached_input_tokens cannot exceed input_tokens — cached tokens are a "
                "subset of the input, not an addition to it. Misreading this is how "
                "cost reports end up double-counting the prompt."
            )


def token_rates(
    provider: str, model: str, *, rates: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Return THB per 1M input/output tokens, refusing to invent missing prices."""
    if rates is None:
        raise KeyError(
            f"No verified token rates supplied for {provider}/{model}. Copy the exact "
            "model's current regional input/output prices into the report command."
        )
    rate_in, rate_out = (float(value) for value in rates)
    if rate_in < 0 or rate_out < 0:
        raise ValueError("token rates must not be negative")
    return rate_in, rate_out


def request_cost(
    provider: str, model: str, usage: Usage, *, rates: tuple[float, float] | None = None,
    cache_read_factor: float | None = None,
) -> float:
    """THB for a single request, cache discount included."""
    rate_in, rate_out = token_rates(provider, model, rates=rates)
    if usage.cached_input_tokens:
        if cache_read_factor is None:
            raise ValueError("provide the provider's cached-input price factor for cached usage")
        if not 0 <= cache_read_factor < 1:
            raise ValueError("cache_read_factor must be a discount in [0, 1)")
    fresh_input = usage.input_tokens - usage.cached_input_tokens
    thb = (
        fresh_input * rate_in
        + usage.cached_input_tokens * rate_in * (cache_read_factor or 0.0)
        + usage.output_tokens * rate_out
    )
    return thb / 1_000_000


def cost_per_1k_requests(
    provider: str, model: str, usage: Usage, *, rates: tuple[float, float] | None = None,
    cache_read_factor: float | None = None,
) -> float:
    """THB per 1,000 requests at this average usage.

    The unit the rest of the course reports in, so an LLM path and a classifier path can
    be compared on the same axis in your cost report.
    """
    return request_cost(
        provider, model, usage, rates=rates, cache_read_factor=cache_read_factor
    ) * 1000


def monthly_thb(
    provider: str, model: str, usage: Usage, requests_per_day: float,
    *, rates: tuple[float, float] | None = None, cache_read_factor: float | None = None,
) -> float:
    """THB per 30-day month at a steady request rate."""
    if requests_per_day < 0:
        raise ValueError("requests_per_day must not be negative")
    return request_cost(
        provider, model, usage, rates=rates, cache_read_factor=cache_read_factor
    ) * requests_per_day * 30


def output_cap_saving(
    provider: str, model: str, usage: Usage, capped_output: int,
    *, rates: tuple[float, float] | None = None, cache_read_factor: float | None = None,
) -> float:
    """THB per 1,000 requests saved by capping output length.

    Compare this against a real provider run. A cap changes output length and may truncate
    answers, so the golden-set gate must still pass before calling it an optimisation.
    """
    if capped_output < 0:
        raise ValueError("capped_output must not be negative")
    if capped_output >= usage.output_tokens:
        return 0.0
    capped = Usage(usage.input_tokens, capped_output, usage.cached_input_tokens)
    return cost_per_1k_requests(
        provider, model, usage, rates=rates, cache_read_factor=cache_read_factor
    ) - cost_per_1k_requests(
        provider, model, capped, rates=rates, cache_read_factor=cache_read_factor
    )


def cache_breakeven_hit_rate(
    provider: str, model: str, prefix_tokens: int, *,
    cache_read_factor: float,
    cache_write_factor: float,
) -> float:
    """Cache hit rate above which prompt caching starts paying for itself.

    With the supplied provider factors, return the hit rate above which caching saves
    money. Report it alongside provider-measured cached tokens, not fixture data.
    """
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    if not provider or not model:
        raise ValueError("provider and model are required to identify cache pricing")
    if (not 0 <= cache_read_factor < 1 or not cache_write_factor >= 1
            or cache_write_factor == float("inf")):
        raise ValueError("cache factors must describe a discount and a finite write premium")
    saving_per_hit = 1.0 - cache_read_factor
    extra_per_miss = cache_write_factor - 1.0
    return extra_per_miss / (saving_per_hit + extra_per_miss)
