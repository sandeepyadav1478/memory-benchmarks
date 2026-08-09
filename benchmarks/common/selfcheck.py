"""
Self-check for token accounting and cost metrics
================================================

No test framework — the repo has none and this needs no new dependency.

    python -m benchmarks.common.selfcheck
"""

from __future__ import annotations

import asyncio

from benchmarks.common.llm_client import _record_usage, get_usage, start_usage
from benchmarks.common.metrics import _percentile, compute_cost_metrics


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeAnthropicUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.input_tokens = prompt
        self.output_tokens = completion


class _FakeResp:
    def __init__(self, usage: object) -> None:
        self.usage = usage


def check_percentile() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([5], 50) == 5
    # nearest-rank: p50 of 1..10 is the 5th value, p95 is the 10th
    ten = list(range(1, 11))
    assert _percentile(ten, 50) == 5, _percentile(ten, 50)
    assert _percentile(ten, 95) == 10, _percentile(ten, 95)
    assert _percentile(ten, 100) == 10
    # unsorted input must not change the answer
    assert _percentile(list(reversed(ten)), 50) == 5
    print("ok  percentile")


def check_provider_field_names() -> None:
    start_usage()
    _record_usage(_FakeResp(_FakeUsage(100, 10)))
    _record_usage(_FakeResp(_FakeAnthropicUsage(50, 5)))
    _record_usage(_FakeResp(None))  # a response without usage must not explode
    u = get_usage()
    assert u == {"prompt_tokens": 150, "completion_tokens": 15, "llm_calls": 2}, u
    print("ok  openai + anthropic field names, missing usage tolerated")


def check_untracked_is_zero_not_crash() -> None:
    # A call made outside an eval item must be silently ignored, not counted
    # against whichever item happens to be running.
    import benchmarks.common.llm_client as lc

    lc._usage.set(None)
    _record_usage(_FakeResp(_FakeUsage(999, 999)))
    assert get_usage() == {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
    print("ok  untracked calls are not billed to anyone")


async def check_concurrent_isolation() -> None:
    """The real risk: one shared LLMClient, max_workers concurrent items."""

    async def item(n: int) -> dict:
        start_usage()
        for _ in range(n):
            await asyncio.sleep(0)  # force interleaving between records
            _record_usage(_FakeResp(_FakeUsage(10, 1)))
        return get_usage()

    results = await asyncio.gather(*(item(n) for n in range(1, 11)))
    for n, got in enumerate(results, start=1):
        assert got["prompt_tokens"] == 10 * n, (n, got)
        assert got["llm_calls"] == n, (n, got)
    print("ok  concurrent items do not mix tokens")


def check_cost_metrics() -> None:
    evaluations = [
        {
            "retrieval": {"search_latency_ms": ms},
            "cutoff_results": {"top_200": {"prompt_tokens": tok, "completion_tokens": 10}},
        }
        for ms, tok in zip([100, 200, 300, 400, 500], [1000, 2000, 3000, 4000, 5000])
    ]
    m = compute_cost_metrics(evaluations, cutoff_label="top_200")
    assert m["search_latency_p50_ms"] == 300, m
    assert m["search_latency_p95_ms"] == 500, m
    assert m["search_latency_mean_ms"] == 300.0, m
    assert m["mean_prompt_tokens"] == 3000.0, m
    assert m["n_token_samples"] == 5, m

    # A run made before this change carries no token fields; it must still report
    # latency rather than failing or inventing zeros.
    legacy = [{"retrieval": {"search_latency_ms": 100}}]
    m2 = compute_cost_metrics(legacy, cutoff_label="top_200")
    assert "mean_prompt_tokens" not in m2, m2
    assert m2["search_latency_p50_ms"] == 100, m2
    print("ok  cost metrics, and legacy results still aggregate")


def main() -> None:
    check_percentile()
    check_provider_field_names()
    check_untracked_is_zero_not_crash()
    asyncio.run(check_concurrent_isolation())
    check_cost_metrics()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
