# Judge audit

Measures what the LOCOMO judge **accepts**, using the benchmark's own judge
prompt and its own client. Nothing here reimplements the judge —
`get_judge_prompt`, `JUDGE_SYSTEM_PROMPT` and `LLMClient` are imported from
`benchmarks/`, so what is measured is the shipped article.

```bash
python -m benchmarks.audit.selfcheck                      # no API key needed
python -m benchmarks.audit.judge_audit --dry-run --limit 60 --show 3
python -m benchmarks.audit.judge_audit --limit 400 --model gpt-4o-mini
```

## Why

An accuracy score is a statement about the judge as much as about the system
under test. The LOCOMO judge prompt lists **seven rules that widen acceptance**
— partial credit, paraphrase, extra detail, date tolerance, semantic overlap,
same referent, knowledge-not-wording — against **two grounds for rejection**,
and states the asymmetry in its own words:

> Use evidence only to ACCEPT answers, never to reject them more strictly.

That is a defensible design choice for semantic recall: "chocolate raspberry
tart" really should match "chocolate cake with raspberries". It is also
unmeasured. This package measures its cost.

## Method

Published results already contain (question, gold answer, generated answer,
verdict) for every item, so the audit re-judges — it never runs mem0 and never
generates answers. Each generated answer is perturbed into one that is **wrong
by construction**, and a CORRECT verdict on it is a false accept.

| perturbation | change | rule it probes |
|---|---|---|
| `identity` | none — positive control | — |
| `transplant` | answer from a different conversation *and* category | "completely different topic", one of the two stated WRONG grounds |
| `abstain` | replaced with a refusal | gold is always factual in categories 1–4 |
| `shift_dates` | every year +2 (730 days) | rule 4, "dates within 14 days" |
| `scale_numbers` | integers ×10, years excluded | rule 4, "durations within 50%" |
| `swap_entity` | subject renamed to someone from another conversation | rule 6, "same referent" |

Perturbations return `None` when they cannot make a guaranteed-wrong change —
no date to shift, no name to swap. Those items are excluded from the
denominator rather than scored, so each rate is against the population that
actually received the treatment.

**Failed calls are held out of every rate.** `locomo/run.py` collapses a
malformed judge response into `WRONG` (`correct = False` on a non-dict). That is
safe for scoring but would be fatal here: a timeout would be indistinguishable
from the judge correctly rejecting an adversarial answer, and would silently
flatter it. `NO_VERDICT` is its own outcome and is reported separately.

It is `NO_VERDICT`, not `PARSE_FAIL`, because the client returns `{}` after
exhausting retries — identically for a rate limit, a timeout and genuinely
malformed output. Calling it a parse failure would name a cause the data does
not distinguish. **A run with a large `no_verdict` count is one to repeat, not
to interpret**: the excluded items are not a random sample, since a judge is
likeliest to stall on the longest prompts.

## Ceilings

Read these before quoting any number.

- **The published runs judged with `gpt-5`.** Any other judge model measures how
  *this prompt* behaves under *that* model. That is a claim about the prompt's
  design, not a reproduction of mem0's published scores, and the two must never
  be stated as if they were the same thing. The report prints the model for
  exactly this reason.
- **`--judge-label` exists because a gateway alias is not a model.** When
  `--model` is a routing name, pass the true backing model; the report records
  that and keeps the alias in a separate field.
- **The control measures cross-model agreement, not self-consistency.** Comparing
  a re-judged verdict against a stored `gpt-5` verdict conflates two models with
  two runs. Run against the same model as the stored verdicts to get a genuine
  consistency figure.
- **A perturbation is an assumption.** `selfcheck.py` asserts each one changes
  what it claims to change, but "wrong by construction" is an argument, not a
  measurement. The dry run prints what gets sent so the argument can be checked
  by eye.
- **mem0 discloses two relevant limits itself** and they should travel with any
  criticism: scores reflect the managed platform "which includes proprietary
  optimizations not available in the open-source SDK", and carry "a ±1 point
  confidence interval due to judge inconsistency".
