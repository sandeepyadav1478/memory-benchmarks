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

## Retrieval-depth churn

```bash
python -m benchmarks.audit.depth_churn      # no API key, reads committed results only
```

Every platform benchmark ships at two retrieval depths over the same questions,
and the reported difference between them is a **net** figure. Net movement and
total movement are different quantities, and here they differ by up to 4.9x:

| benchmark | n | top_50 | top_200 | net | items that flip | gained | regressed |
|---|---|---|---|---|---|---|---|
| locomo | 1539 | 82.66% | 91.56% | +8.90 | 191 (12.41%) | 164 | 27 |
| longmemeval | 500 | 90.40% | 93.40% | +3.00 | 35 (7.00%) | 25 | 10 |
| beam_1m | 700 | 67.14% | 70.14% | +3.00 | 103 (14.71%) | 62 | 41 |
| beam_10m | 200 | 45.50% | 50.50% | +5.00 | 38 (19.00%) | 24 | 14 |

`beam_1m` is the clearest case: a +3.00 point gain is **62 improvements against
41 regressions**. The headline is accurate and it describes two fifths less
movement than actually happened.

**Deeper retrieval is not uniformly better, and where it hurts is explainable.**
`contradiction_resolution` is the worst-regressing category in both BEAM runs —
30.0% on `beam_10m`, 14.3% on `beam_1m` — which is what you would predict:
retrieving more memories surfaces more mutually inconsistent ones, and
reconciling them *is* the task. On LOCOMO the regressions concentrate in
`open-domain` (4.2%) and `temporal` (2.5%).

Checked and **not** the explanation: BEAM's correctness is a threshold
(`score >= 0.5`) on a continuous rubric score, so flips could have been the cut
wobbling under trivial score changes. They are not — only 1.9% (`beam_1m`) and
0% (`beam_10m`) of flips involve a score move of 0.1 or less.

### Judge self-consistency

Both LOCOMO runs record `with_evidence: False`, and `get_judge_prompt` is a pure
function of (question, gold, answer) — it takes `category` and does not use it.
So an item whose generated answer is byte-identical across the two runs gave the
judge a byte-identical prompt.

**407 such items on LOCOMO; 1 changed verdict (0.25%).** That is
`conv0_q68` — gold "Since 2016", answer "Seven years." in both runs, CORRECT at
top_50 and WRONG at top_200. BEAM shows 0 of 69 and 0 of 37.

This **corroborates** mem0's own disclosure of "a ±1 point confidence interval
due to judge inconsistency" rather than disputing it. Quote it that way.

## Ceilings

Read these before quoting any number.

- **`depth_churn.py` asserts its recomputation against each file's own
  `metrics_by_cutoff` before reporting anything.** If a verdict convention is
  misread the run dies rather than publishing a number. That is the only reason
  these figures can be trusted without re-judging anything.
- **BEAM's `>= 0.5` correctness threshold is derived, not documented.** It is
  justified solely by reproducing all four published `correct` counts exactly
  (491, 470, 101, 91). If mem0 uses a different rule that happens to agree on
  these four files, the churn figures move.
- **Churn is not error.** An item flipping between depths is the expected
  behaviour of a retrieval change. The claim here is only that the net figure
  is a lossy summary of it, and that the regressions have structure.

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
