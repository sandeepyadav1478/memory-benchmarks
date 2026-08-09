"""
Judge Quality Audit
===================

Measures what the LOCOMO judge accepts, using the benchmark's own judge prompt
and its own client. Nothing here reimplements the judge; ``get_judge_prompt``
and ``LLMClient`` are imported from the benchmark so that what is measured is
the shipped article.

Two questions, one pipeline:

*Leniency* -- feed the judge answers that are wrong by construction and count
how many it marks CORRECT. The judge prompt lists seven rules that widen
acceptance (partial credit, paraphrase, extra detail, date tolerance, semantic
overlap, same referent, knowledge-not-wording) against two grounds for
rejection, and states its bias in its own words: evidence may be used "only to
ACCEPT answers, never to reject them more strictly". This measures the cost of
that choice.

*Self-consistency* -- re-judge unmodified answers and compare against the
verdict already stored in the results file. mem0's documentation states scores
"carry a +/-1 point confidence interval due to judge inconsistency", which is a
falsifiable claim about exactly this number.

    python -m benchmarks.audit.judge_audit --limit 50 --model claude-haiku-4-5 \
        --base-url http://localhost:4000/v1

⚠️ The published runs judged with ``gpt-5``. Any other model measures how *this
prompt* behaves under *that* model, which is a claim about the prompt's design,
not a reproduction of mem0's published scores. The report prints the model used
for exactly this reason; quote it whenever the numbers are quoted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.audit.perturb import PERTURBATIONS, build, pick_donor
from benchmarks.common.llm_client import LLMClient
from benchmarks.locomo.prompts import (
    CATEGORIES_TO_EVALUATE,
    JUDGE_SYSTEM_PROMPT,
    get_judge_prompt,
    preprocess_answer,
)

DEFAULT_RESULTS = "results/platform/locomo_results.json"

# A verdict the judge should give if the perturbation did its job. identity is
# the control, where agreeing with the stored verdict is the desired outcome.
EXPECT_CORRECT = {"identity"}


def load_items(path: str, cutoff: str | None = None) -> list[dict]:
    """Flatten a published results file into judge inputs.

    Only items the benchmark itself scores are returned: categories 1-4, with a
    non-empty generated answer and a gold answer to judge against.
    """
    data = json.loads(Path(path).read_text())
    items = []
    for e in data["evaluations"]:
        cutoffs = e.get("cutoff_results") or {}
        label = cutoff or (sorted(cutoffs)[-1] if cutoffs else None)
        cr = cutoffs.get(label) or {}
        gold = e.get("ground_truth_answer")
        answer = (cr.get("generated_answer") or "").strip()
        category = e.get("category")
        if category not in CATEGORIES_TO_EVALUATE or not answer or not gold:
            continue
        items.append(
            {
                "question_id": e.get("question_id"),
                "conversation_idx": e.get("conversation_idx"),
                "category": category,
                "category_name": e.get("category_name", ""),
                "question": e["question"],
                # The benchmark trims category-3 gold answers before judging;
                # judging against the untrimmed string would not be its judge.
                "gold": preprocess_answer(category, str(gold)),
                "generated_answer": answer,
                "published_judgment": cr.get("judgment"),
                "cutoff": label,
            }
        )
    return items


def stratified_sample(items: list[dict], limit: int, rng: random.Random) -> list[dict]:
    """Sample within category so every category keeps its share of the budget.

    A flat random sample would be dominated by single-hop, which is 55% of
    LOCOMO, and the per-category rates are the point.
    """
    if limit <= 0 or limit >= len(items):
        return list(items)
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    out: list[dict] = []
    for cat in sorted(by_cat):
        share = max(1, round(limit * len(by_cat[cat]) / len(items)))
        out.extend(rng.sample(by_cat[cat], min(share, len(by_cat[cat]))))
    return out


async def judge_once(llm: LLMClient, item: dict, answer: str) -> tuple[str, str]:
    """Run the benchmark's judge on one answer.

    Returns (verdict, reasoning) where verdict is CORRECT, WRONG or PARSE_FAIL.

    The benchmark collapses a malformed judge response into WRONG
    (``locomo/run.py``: ``correct = False`` when the parse yields a non-dict).
    That is safe for scoring but not for this audit -- a parse failure would be
    indistinguishable from the judge correctly rejecting an adversarial answer,
    which would silently flatter the judge. So it is kept as its own outcome and
    excluded from the rates below.
    """
    prompt = get_judge_prompt(item["category"], item["question"], item["gold"], answer)
    try:
        raw = await llm.generate_structured(system=JUDGE_SYSTEM_PROMPT, user=prompt)
    except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
        return "PARSE_FAIL", f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, dict) or "label" not in raw:
        return "PARSE_FAIL", str(raw)[:200]
    label = str(raw.get("label", "")).upper()
    reasoning = str(raw.get("reasoning", ""))[:300]
    if label not in {"CORRECT", "WRONG"}:
        return "PARSE_FAIL", f"label={label!r}"
    return label, reasoning


async def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    all_items = load_items(args.results, args.cutoff)
    items = stratified_sample(all_items, args.limit, rng)
    print(f"{len(all_items)} scorable items, {len(items)} sampled (seed {args.seed})")

    # Names from other conversations, for entity swaps. Built once from the full
    # set so the pool does not shrink with the sample.
    from benchmarks.audit.perturb import _names

    name_pool = sorted({n for it in all_items for n in _names(it["question"])})

    llm = LLMClient(
        model=args.model,
        provider=args.provider,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url,
        rpm=args.rpm,
    )

    kinds = args.perturbations or list(PERTURBATIONS)
    jobs = []
    for i, it in enumerate(items):
        donor = pick_donor(items, i, rng)
        foreign = [n for n in name_pool if n not in it["question"]]
        for kind in kinds:
            answer = build(kind, it, donor, foreign)
            if answer is not None:
                jobs.append((kind, it, answer))

    print(f"{len(jobs)} judge calls across {len(kinds)} perturbations")
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(kind: str, it: dict, answer: str) -> dict:
        nonlocal done
        async with sem:
            verdict, reasoning = await judge_once(llm, it, answer)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(jobs)}", flush=True)
        return {
            "perturbation": kind,
            "question_id": it["question_id"],
            "category": it["category"],
            "category_name": it["category_name"],
            "question": it["question"],
            "gold": it["gold"],
            "judged_answer": answer,
            "verdict": verdict,
            "reasoning": reasoning,
            "published_judgment": it["published_judgment"],
        }

    records = await asyncio.gather(*(one(k, i, a) for k, i, a in jobs))
    return summarise(records, args)


def summarise(records: list[dict], args: argparse.Namespace) -> dict:
    """Rates per perturbation, with parse failures held out of the denominator."""
    by_kind: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_kind[r["perturbation"]][r["verdict"]] += 1

    rows = {}
    for kind, c in by_kind.items():
        judged = c["CORRECT"] + c["WRONG"]
        rows[kind] = {
            "n_attempted": sum(c.values()),
            "n_judged": judged,
            "parse_failures": c["PARSE_FAIL"],
            "accepted": c["CORRECT"],
            "accept_rate_pct": round(c["CORRECT"] / judged * 100, 2) if judged else None,
        }

    # Control: does re-judging an unchanged answer reproduce the stored verdict?
    ident = [r for r in records if r["perturbation"] == "identity" and r["verdict"] != "PARSE_FAIL"]
    agree = sum(1 for r in ident if r["verdict"] == r["published_judgment"])
    consistency = {
        "n": len(ident),
        "agreed_with_published": agree,
        "flip_rate_pct": round((len(ident) - agree) / len(ident) * 100, 2) if ident else None,
        "note": (
            "Published verdicts were produced by gpt-5; a different judge model "
            "measures cross-model agreement, not run-to-run self-consistency."
        ),
    }

    return {
        "config": {
            "results_file": args.results,
            "judge_model": args.judge_label or args.model,
            "judge_route": args.model if args.judge_label else None,
            "base_url": args.base_url,
            "published_judge_model": "gpt-5",
            "seed": args.seed,
            "concurrency": args.concurrency,
        },
        "leniency": rows,
        "consistency": consistency,
        "records": records,
    }


def report(summary: dict) -> None:
    cfg = summary["config"]
    print("\n" + "=" * 72)
    print(f"judge under test : {cfg['judge_model']}   (published runs used {cfg['published_judge_model']})")
    print("=" * 72)
    print(f"{'perturbation':<16}{'judged':>8}{'accepted':>10}{'accept %':>10}{'parse fail':>12}")
    for kind in PERTURBATIONS:
        r = summary["leniency"].get(kind)
        if not r:
            continue
        expected = "control" if kind in EXPECT_CORRECT else "false accepts"
        rate = "n/a" if r["accept_rate_pct"] is None else f"{r['accept_rate_pct']:.1f}"
        print(f"{kind:<16}{r['n_judged']:>8}{r['accepted']:>10}{rate:>10}{r['parse_failures']:>12}   {expected}")
    c = summary["consistency"]
    print(f"\ncontrol agreement with stored verdicts: {c['agreed_with_published']}/{c['n']} "
          f"(flip rate {c['flip_rate_pct']}%)")
    print(f"note: {c['note']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Audit the LOCOMO judge for false accepts")
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--cutoff", default=None, help="Cutoff label (default: largest present)")
    p.add_argument("--limit", type=int, default=50, help="Questions to sample (0 = all)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument(
        "--judge-label",
        default=None,
        help="True model behind --model, when that is a gateway alias. Recorded in "
             "the report instead of the alias: a proxy alias is a routing name, "
             "not the identity of the model that produced the verdicts.",
    )
    p.add_argument("--provider", default="openai")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    p.add_argument("--api-key", default=None)
    p.add_argument("--rpm", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturbations", nargs="*", choices=PERTURBATIONS, default=None)
    p.add_argument("--out", default="results/audit/judge_audit.json")
    p.add_argument("--dry-run", action="store_true", help="Print perturbed answers, call nothing")
    p.add_argument("--show", type=int, default=2, help="Items to print in full during --dry-run")
    args = p.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    summary = asyncio.run(run(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    report(summary)
    print(f"\nwrote {out}")


def dry_run(args: argparse.Namespace) -> None:
    """Show what each perturbation produces, without an API key.

    The perturbations are the part that has to be right -- a probe that is not
    actually wrong makes every number downstream meaningless -- so they are
    reviewable on their own.
    """
    rng = random.Random(args.seed)
    items = stratified_sample(load_items(args.results, args.cutoff), args.limit, rng)
    from benchmarks.audit.perturb import _names

    name_pool = sorted({n for it in items for n in _names(it["question"])})
    counts: Counter = Counter()
    for i, it in enumerate(items):
        donor = pick_donor(items, i, rng)
        foreign = [n for n in name_pool if n not in it["question"]]
        for kind in args.perturbations or list(PERTURBATIONS):
            answer = build(kind, it, donor, foreign)
            counts[kind] += answer is not None
            if answer is not None and i < args.show:
                print(f"\n[{kind}] {it['category_name']} | {it['question'][:80]}")
                print(f"  gold : {it['gold'][:100]}")
                print(f"  orig : {it['generated_answer'][:100]}")
                print(f"  sent : {answer[:100]}")
    print("\napplicable / sampled:")
    for kind in args.perturbations or list(PERTURBATIONS):
        print(f"  {kind:<16}{counts[kind]:>5} / {len(items)}")


if __name__ == "__main__":
    main()
