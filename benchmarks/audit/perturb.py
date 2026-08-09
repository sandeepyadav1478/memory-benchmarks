"""
Adversarial perturbations for judge auditing
============================================

Each perturbation turns a generated answer into one that is **wrong by
construction**, so that a CORRECT verdict from the judge is a false accept
rather than a debatable call. That property is the whole experiment: a
perturbation that might still be right measures nothing.

Every function returns ``None`` when it cannot make a guaranteed-wrong change
(no date to shift, no name to swap). Callers must exclude those items from the
denominator instead of scoring them, or the reported rate is against a
population that did not receive the treatment.
"""

from __future__ import annotations

import random
import re

# A year we can shift by two, well outside the judge's stated 14-day tolerance
# and its "durations within 50%" rule.
_YEAR = re.compile(r"\b(20(?:1[5-9]|2[0-5]))\b")
_INT = re.compile(r"\b([1-9]\d{0,3})\b")
_NAME = re.compile(r"\b([A-Z][a-z]{2,11})\b")

# Words that look like names to the regex but are sentence-initial or generic.
_NOT_NAMES = frozenset(
    """The This That These Those There Their They Then When What Which Where While Who Whose
    Her His She Him Its And But For From With Without During After Before Both Each Every
    All Any One Two Three Four Five Six Seven Eight Nine Ten Yes Not None Some Both
    January February March April May June July August September October November December
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday
    User Answer Question Memory Memories Based According Note Also However Additionally
    Sorry Unfortunately""".split()
)

ABSTENTION = (
    "I don't have enough information in the retrieved memories to answer that question."
)


def abstain(answer: str) -> str | None:
    """Replace the answer with a refusal.

    LOCOMO categories 1-4 all have a factual gold answer, so a refusal is always
    wrong. Category 5 (adversarial), where refusing is correct, is excluded from
    scoring by the benchmark itself.
    """
    return ABSTENTION if answer.strip() else None


def shift_dates(answer: str, years: int = 2) -> str | None:
    """Move every year in the answer forward, leaving the rest of the text alone.

    Two years is 730 days against a stated tolerance of 14, so the result cannot
    be salvaged by the date rule. Returns None when the answer states no year --
    those items get no date treatment and must not be counted as passing it.
    """
    if not _YEAR.search(answer):
        return None
    out = _YEAR.sub(lambda m: str(int(m.group(1)) + years), answer)
    return out if out != answer else None


def scale_numbers(answer: str, factor: int = 10) -> str | None:
    """Multiply standalone integers, skipping any that are part of a date.

    The judge accepts numbers "within 50%" and off-by-one durations; a factor of
    ten clears both. Years are left to shift_dates so the two perturbations stay
    independent and a failure can be attributed to one rule.
    """
    spans = [m.span() for m in _YEAR.finditer(answer)]

    def repl(m: re.Match) -> str:
        s, e = m.span()
        if any(s >= a and e <= b for a, b in spans):
            return m.group(0)
        return str(int(m.group(1)) * factor)

    out = _INT.sub(repl, answer)
    return out if out != answer else None


def _names(text: str) -> list[str]:
    """Candidate person names, in order of appearance, deduplicated."""
    seen: dict[str, None] = {}
    for m in _NAME.finditer(text):
        w = m.group(1)
        if w not in _NOT_NAMES:
            seen.setdefault(w, None)
    return list(seen)


def swap_entity(answer: str, question: str, gold: str, foreign_names: list[str]) -> str | None:
    """Rename the subject of the answer to someone from another conversation.

    Rule 6 of the judge prompt makes "the same core entity" sufficient for
    CORRECT, so changing the entity should be sufficient for WRONG. The
    replacement is drawn from a different conversation and checked against the
    question and gold answer, because a name that appears in the question would
    leave the answer arguably still about the right person.
    """
    protected = set(_names(question)) | set(_names(gold))
    targets = [n for n in _names(answer) if n not in protected]
    pool = [n for n in foreign_names if n not in protected and n not in targets]
    if not targets or not pool:
        return None
    out = re.sub(rf"\b{re.escape(targets[0])}\b", pool[0], answer)
    return out if out != answer else None


def transplant(foreign_answer: str) -> str | None:
    """Answer a different question entirely.

    The judge prompt names this as one of its only two grounds for WRONG --
    "the answer addresses a completely different topic" -- so it is the cleanest
    possible probe, and the one whose expected verdict is least arguable.
    """
    return foreign_answer.strip() or None


PERTURBATIONS = ("identity", "transplant", "abstain", "shift_dates", "scale_numbers", "swap_entity")


def build(kind: str, item: dict, donor: dict, foreign_names: list[str]) -> str | None:
    """Apply one perturbation to an item, or return None if it does not apply.

    ``identity`` is the positive control: the unmodified answer, re-judged. Its
    verdict measures judge self-consistency rather than leniency, which is a
    different question that happens to need the same plumbing.
    """
    answer = item["generated_answer"]
    if kind == "identity":
        return answer
    if kind == "transplant":
        return transplant(donor["generated_answer"])
    if kind == "abstain":
        return abstain(answer)
    if kind == "shift_dates":
        return shift_dates(answer)
    if kind == "scale_numbers":
        return scale_numbers(answer)
    if kind == "swap_entity":
        return swap_entity(answer, item["question"], item["gold"], foreign_names)
    raise ValueError(f"unknown perturbation: {kind}")


def pick_donor(items: list[dict], idx: int, rng: random.Random) -> dict:
    """Choose a transplant source from a different conversation and category.

    Both constraints matter: same-conversation answers share people and places,
    and same-category answers share shape, either of which could make a
    transplanted answer accidentally defensible.
    """
    here = items[idx]
    pool = [
        it
        for it in items
        if it["conversation_idx"] != here["conversation_idx"]
        and it["category"] != here["category"]
        and it["generated_answer"].strip()
    ]
    return rng.choice(pool) if pool else items[(idx + len(items) // 2) % len(items)]
