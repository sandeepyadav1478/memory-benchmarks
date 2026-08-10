"""What changes when retrieval depth changes, measured per item.

Every published platform benchmark ships twice -- once at ``top_200`` and once
at ``top_50`` -- over the same question set. The README reports the difference
between the two accuracies. That difference is a *net* figure, and a net figure
cannot distinguish "50 items got better" from "150 got better and 100 got
worse". This module recomputes both, per item.

It reads only committed result files. It never calls a model, never runs mem0
and never re-judges anything, so it costs nothing to run and cannot disagree
with the published numbers for any reason except arithmetic -- which is exactly
why every recomputed accuracy here is asserted against the file's own
``metrics_by_cutoff`` block before it is used.

    python -m benchmarks.audit.depth_churn

Three benchmarks, three verdict conventions:

* LOCOMO      ``judgment`` is ``CORRECT`` / ``WRONG`` / ``ERROR``.
* LongMemEval ``judgment`` is ``PASS`` / ``FAIL``.
* BEAM        no ``judgment`` at all -- a continuous rubric ``score``, made
  binary at ``>= 0.5``. That threshold is not documented; it is *derived*, and
  the assertion against the published ``correct`` counts is what justifies it.
  All four BEAM files reproduce exactly, which is the only evidence offered
  that the threshold is right.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "platform"

BEAM_CORRECT_AT = 0.5

# (name, deep file, shallow file, category field or None)
PAIRS = [
    ("locomo", "locomo_results.json", "locomo_top50_results.json", "category_name"),
    ("longmemeval", "longmemeval_results.json", "longmemeval_top50_results.json", "question_type"),
    ("beam_1m", "beam_1m_results.json", "beam_1m_top50_results.json", "question_type"),
    ("beam_10m", "beam_10m_results.json", "beam_10m_top50_results.json", "question_type"),
]


def verdict(cut: dict) -> bool | None:
    """True/False for a scorable item, None for one that cannot be scored.

    None is returned for LOCOMO's ``ERROR`` judgment -- a harness failure, not a
    wrong answer. Folding it into False would make an infrastructure error look
    like a model mistake, and it would land in the churn count as a flip.
    """
    j = cut.get("judgment")
    if j in ("CORRECT", "PASS"):
        return True
    if j in ("WRONG", "FAIL"):
        return False
    if j == "ERROR" or j is None and "score" not in cut:
        return None
    if j is None:
        return cut["score"] >= BEAM_CORRECT_AT
    return None


@dataclass
class Churn:
    name: str
    n: int
    deep_acc: float
    shallow_acc: float
    gained: int
    regressed: int
    by_category: dict
    same_answer_n: int
    same_answer_flips: list

    @property
    def net(self) -> float:
        return self.deep_acc - self.shallow_acc

    @property
    def flips(self) -> int:
        return self.gained + self.regressed

    @property
    def churn_pct(self) -> float:
        return 100.0 * self.flips / self.n


def load(fname: str) -> tuple[list, str, dict]:
    d = json.loads((RESULTS / fname).read_text())
    label = d["metadata"]["top_k_cutoffs"][0]
    return d["evaluations"], label, d["metrics_by_cutoff"][label]["overall"]


def published_correct(overall: dict) -> int:
    # LongMemEval's deep file says "passed"; everything else says "correct".
    return overall.get("correct", overall.get("passed"))


def analyse(name: str, deep_f: str, shallow_f: str, cat_field: str | None) -> Churn:
    deep_ev, deep_lbl, deep_pub = load(deep_f)
    shal_ev, shal_lbl, shal_pub = load(shallow_f)

    deep = {it["question_id"]: it for it in deep_ev}
    shal = {it["question_id"]: it for it in shal_ev}

    # Recompute the published accuracy and refuse to continue if it disagrees.
    # This is the whole safety net: if verdict() misreads a convention, the
    # totals stop matching and the run dies here rather than shipping a number.
    for ev, lbl, pub in ((deep_ev, deep_lbl, deep_pub), (shal_ev, shal_lbl, shal_pub)):
        mine = sum(1 for it in ev if verdict(it["cutoff_results"][lbl]) is True)
        assert mine == published_correct(pub), (
            f"{name}/{lbl}: recomputed {mine} correct, file publishes "
            f"{published_correct(pub)} -- verdict() misreads this convention"
        )

    common = deep.keys() & shal.keys()
    gained = regressed = 0
    scored = 0
    by_cat: dict[str, Counter] = {}
    same_answer_n = 0
    same_answer_flips = []

    for qid in common:
        dv = verdict(deep[qid]["cutoff_results"][deep_lbl])
        sv = verdict(shal[qid]["cutoff_results"][shal_lbl])
        if dv is None or sv is None:
            continue  # a ratio needs a denominator that was actually treated
        scored += 1
        cat = deep[qid].get(cat_field, "?") if cat_field else "?"
        c = by_cat.setdefault(cat, Counter())
        c["n"] += 1
        if dv != sv:
            c["flip"] += 1
            if dv:
                gained += 1
            else:
                regressed += 1
                c["regress"] += 1

        # Judge self-consistency: identical answer text, so identical judge
        # input -- both runs record with_evidence=False and get_judge_prompt
        # is a pure function of (question, gold, answer).
        da = deep[qid]["cutoff_results"][deep_lbl].get("generated_answer")
        sa = shal[qid]["cutoff_results"][shal_lbl].get("generated_answer")
        if da is not None and da == sa:
            same_answer_n += 1
            if dv != sv:
                same_answer_flips.append(
                    {
                        "question_id": qid,
                        "question": deep[qid].get("question", "")[:90],
                        "gold": str(deep[qid].get("ground_truth_answer", ""))[:60],
                        "answer": str(da)[:60],
                        "deep": dv,
                        "shallow": sv,
                    }
                )

    return Churn(
        name=name,
        n=scored,
        deep_acc=100.0 * published_correct(deep_pub) / deep_pub["total"],
        shallow_acc=100.0 * published_correct(shal_pub) / shal_pub["total"],
        gained=gained,
        regressed=regressed,
        by_category={k: dict(v) for k, v in by_cat.items()},
        same_answer_n=same_answer_n,
        same_answer_flips=same_answer_flips,
    )


def beam_threshold_proximity(band: float = 0.1) -> dict:
    """How many BEAM flips are just the 0.5 cut moving under a small score change.

    BEAM's binary correctness is a threshold on a continuous score, so an item
    scoring 0.49 and one scoring 0.51 are one nugget apart and land on opposite
    sides. This measures how much of BEAM's churn is that, rather than a real
    change in answer quality.
    """
    out = {}
    for name, deep_f, shallow_f, _ in PAIRS:
        if not name.startswith("beam"):
            continue
        deep_ev, dl, _ = load(deep_f)
        shal_ev, sl, _ = load(shallow_f)
        deep = {i["question_id"]: i["cutoff_results"][dl]["score"] for i in deep_ev}
        shal = {i["question_id"]: i["cutoff_results"][sl]["score"] for i in shal_ev}
        flips = near = 0
        for q in deep.keys() & shal.keys():
            if (deep[q] >= BEAM_CORRECT_AT) != (shal[q] >= BEAM_CORRECT_AT):
                flips += 1
                if abs(deep[q] - shal[q]) <= band:
                    near += 1
        out[name] = {"flips": flips, "within_band": near, "band": band}
    return out


def main() -> None:
    results = [analyse(*p) for p in PAIRS]

    print(f"{'benchmark':<12} {'n':>5} {'top50':>7} {'top200':>7} {'net':>7} "
          f"{'flips':>6} {'churn':>7} {'gain':>5} {'regress':>8}")
    print("-" * 74)
    for r in results:
        print(f"{r.name:<12} {r.n:>5} {r.shallow_acc:>6.2f}% {r.deep_acc:>6.2f}% "
              f"{r.net:>+6.2f} {r.flips:>6} {r.churn_pct:>6.2f}% {r.gained:>5} {r.regressed:>8}")

    print("\nchurn hidden by the net figure (flips per net point gained):")
    for r in results:
        if abs(r.net) > 0.01:
            moved = 100.0 * r.flips / r.n
            print(f"  {r.name:<12} net {r.net:+.2f} pts, but {moved:.2f}% of items moved "
                  f"-- {moved / abs(r.net):.1f}x the net")

    print("\nregressions by category (deeper retrieval made these worse):")
    for r in results:
        rows = [(c, v) for c, v in r.by_category.items() if v.get("regress")]
        if not rows:
            continue
        print(f"  {r.name}:")
        for cat, v in sorted(rows, key=lambda kv: -kv[1]["regress"]):
            print(f"    {cat:<24} {v['regress']:>3} regressed of {v['n']:>4} "
                  f"({100.0 * v['regress'] / v['n']:.1f}%)")

    print("\njudge self-consistency (identical answer text -> identical judge input):")
    for r in results:
        if not r.same_answer_n:
            continue
        rate = 100.0 * len(r.same_answer_flips) / r.same_answer_n
        print(f"  {r.name:<12} {len(r.same_answer_flips):>3} verdict changes on "
              f"{r.same_answer_n:>4} identical answers ({rate:.2f}%)")
        for f in r.same_answer_flips[:5]:
            print(f"      {f['question_id']}: gold={f['gold']!r} answer={f['answer']!r} "
                  f"top50={'OK' if f['shallow'] else 'X'} top200={'OK' if f['deep'] else 'X'}")

    print("\nBEAM threshold proximity (score>=0.5 is the cut):")
    for name, v in beam_threshold_proximity().items():
        pct = 100.0 * v["within_band"] / v["flips"] if v["flips"] else 0.0
        print(f"  {name:<12} {v['within_band']:>3} of {v['flips']:>3} flips move the score "
              f"by <= {v['band']} ({pct:.1f}%) -- threshold noise, not answer quality")

    # Runnable check. These are the numbers quoted elsewhere; if a result file
    # is ever regenerated they should fail loudly rather than drift silently.
    loc = next(r for r in results if r.name == "locomo")
    assert loc.n == 1539, loc.n                      # 1540 minus the one ERROR item
    assert loc.flips == 191, loc.flips
    assert (loc.gained, loc.regressed) == (164, 27), (loc.gained, loc.regressed)
    assert loc.same_answer_n == 407, loc.same_answer_n
    assert len(loc.same_answer_flips) == 1, loc.same_answer_flips
    print("\nself-check OK")


if __name__ == "__main__":
    main()
