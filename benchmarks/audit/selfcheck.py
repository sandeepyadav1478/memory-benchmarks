"""
Self-check for the judge audit
==============================

    python -m benchmarks.audit.selfcheck

The perturbations carry the whole experiment: if a probe can still be a correct
answer, a CORRECT verdict is not a false accept and every rate is noise. So the
checks below are mostly about the *guarantees* -- that a change was made, that
it was the intended change, and that inapplicable cases return None rather than
quietly passing the original through.
"""

from __future__ import annotations

import random

from benchmarks.audit.perturb import (
    ABSTENTION,
    _names,
    abstain,
    build,
    pick_donor,
    scale_numbers,
    shift_dates,
    swap_entity,
    transplant,
)


def check_dates() -> None:
    assert shift_dates("She joined in May 2023.") == "She joined in May 2025."
    # Every year moves, so a multi-date answer cannot stay half-right.
    assert shift_dates("From 2021 to 2022") == "From 2023 to 2024"
    # No year: no treatment, and the caller must not count this item.
    assert shift_dates("She joined the group.") is None
    # A shift of two years is 730 days against a stated 14-day tolerance.
    assert shift_dates("2019") == "2021"
    print("ok  shift_dates")


def check_numbers() -> None:
    assert scale_numbers("He read 3 books") == "He read 30 books"
    # Years belong to shift_dates; scaling them here would confound the two.
    assert scale_numbers("In 2023 he read 3 books") == "In 2023 he read 30 books"
    assert scale_numbers("no digits here") is None
    print("ok  scale_numbers")


def check_abstain() -> None:
    assert abstain("anything") == ABSTENTION
    assert abstain("   ") is None
    print("ok  abstain")


def check_names() -> None:
    # Sentence-initial and calendar words are not people.
    assert _names("The trip was in May with Caroline and Melanie.") == ["Caroline", "Melanie"]
    assert _names("What did she do?") == []
    print("ok  name extraction")


def check_swap_entity() -> None:
    out = swap_entity("Caroline went hiking.", "What did Melanie do?", "went skiing", ["Tim"])
    assert out == "Tim went hiking.", out

    # A name that appears in the question is the subject under test; renaming it
    # would leave the answer arguably still about the right person, so the swap
    # must decline rather than produce a weak probe.
    assert swap_entity("Caroline went hiking.", "What did Caroline do?", "hiked", ["Tim"]) is None
    # Nothing to swap in.
    assert swap_entity("Caroline went hiking.", "What did she do?", "hiked", []) is None
    # Substring safety: swapping "Tim" must not damage "Timothy".
    out2 = swap_entity("Tim and Timothy left.", "Who left?", "they did", ["Joanna"])
    assert out2 == "Joanna and Timothy left.", out2
    print("ok  swap_entity")


def check_transplant_and_dispatch() -> None:
    assert transplant("  other answer  ") == "other answer"
    assert transplant("   ") is None

    item = {
        "generated_answer": "Caroline went hiking in 2023 with 2 friends.",
        "question": "What did she do?",
        "gold": "hiked",
        "category": 2,
        "conversation_idx": 0,
    }
    donor = {"generated_answer": "He bought a guitar.", "category": 4, "conversation_idx": 1}

    assert build("identity", item, donor, ["Tim"]) == item["generated_answer"]
    assert build("transplant", item, donor, ["Tim"]) == "He bought a guitar."
    assert build("abstain", item, donor, ["Tim"]) == ABSTENTION
    assert "2025" in build("shift_dates", item, donor, ["Tim"])
    assert "20 friends" in build("scale_numbers", item, donor, ["Tim"])
    assert build("swap_entity", item, donor, ["Tim"]).startswith("Tim went")

    # identity must be byte-identical, or the control measures a rewrite rather
    # than the judge.
    assert build("identity", item, donor, ["Tim"]) is item["generated_answer"]
    print("ok  transplant + dispatch")


def check_donor_is_foreign() -> None:
    items = [
        {"conversation_idx": i // 2, "category": 1 + i % 4, "generated_answer": f"answer {i}"}
        for i in range(20)
    ]
    rng = random.Random(0)
    for i, it in enumerate(items):
        d = pick_donor(items, i, rng)
        assert d["conversation_idx"] != it["conversation_idx"], (i, d)
        assert d["category"] != it["category"], (i, d)
    print("ok  donors are cross-conversation and cross-category")


def main() -> None:
    check_dates()
    check_numbers()
    check_abstain()
    check_names()
    check_swap_entity()
    check_transplant_and_dispatch()
    check_donor_is_foreign()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
