# Benchmark protocol (frozen before any result is produced)

This file fixes the design, metrics and what would count as failure **before**
the harness is run on real data. If a choice below changes after seeing
results, record the change and the reason at the bottom; do not edit silently.

## Question

Does CGLC (contract + leases + checkpoint judge + conjunctive gate) reach a
better quality / cost / safety trade-off than simpler controls, on bounded
document-grounded multi-hop QA? (ARCHITECTURE.md §11, ROADMAP #1.)

## Data

HotpotQA (distractor dev), 2WikiMultiHopQA (dev), MuSiQue (answerable dev),
read from the datasets' original local files. Each item's own paragraphs
(gold + distractors) are the closed corpus. Items are sampled with a fixed
seed; items whose corpus exceeds `--max-chars` are dropped so every arm runs
in whole-document mode. The harness never downloads anything itself.

## Arms (§11.1), same worker class, model and corpus for all

`uncontrolled_worker`, `fixed_shallow` (1 action), `fixed_deep` (5 actions),
`generic_prompt` ("be thorough but concise"), `arch3_fixed_checkpoints`
(checkpoint every 3 actions, ordinal rules, no scoring; an approximation of
Architecture 3, not a re-implementation), `cglc_no_stall_trigger`,
`cglc_rule_only`, `cglc_full`, `always_on_judge_upper_bound` (checkpoint after
every action).

## Contract regimes (reported separately, never pooled)

* `oracle_docs`: one hard obligation per **gold supporting paragraph**
  ("this document is cited and relied on"). Leaks which documents matter, not
  their content or the answer. Measures CGLC under a good contract.
* `generic`: one content-free obligation. No benchmark knowledge in the
  contract. This is the honest regime for any claim about real use.

Baselines do not receive obligations and are run once per item.

## Metrics (per item, then paired across arms)

* Answer: EM and token-F1 against gold + aliases, after stripping citation
  brackets. *Correct* := EM or F1 >= 0.5.
* Evidence recall: fraction of gold paragraphs the worker actually cited
  (quotes verified verbatim against the corpus).
* `accepted`: baselines accept their output; CGLC accepts only on
  `ALLOW_FINALIZE`.
* **false-allow** := accepted and not correct. **unsupported-accept** :=
  accepted with evidence recall < 1. **false-block** := not accepted but
  correct (a correct answer withheld).
* Selective F1 := F1 over accepted answers only.
* Cost: worker tokens, controller tokens, tool calls, wall-clock, worker steps.
* Over-execution := worker steps after cited evidence first covered all gold
  paragraphs.

Primary comparisons: `cglc_full` minus each other arm on (a) F1, (b) total
tokens, (c) false-allow rate, as paired differences with 95% bootstrap CIs
over items (2000 resamples). Secondary: everything else in the table.

## Falsification (from ARCHITECTURE.md §11.3, made concrete)

The claim is **not supported** on a dataset/regime if any of these holds:

1. `uncontrolled_worker`, `generic_prompt` or a fixed budget matches
   `cglc_full` on F1 and false-allow (CI of the difference includes 0) at
   equal or lower tokens.
2. Controller tokens are large enough that `cglc_full` total tokens exceed
   `fixed_deep` without a F1 or false-allow gain whose CI excludes 0.
3. `cglc_full` false-allow or unsupported-accept is not lower than
   `uncontrolled_worker`.
4. `cglc_no_stall_trigger` or `cglc_rule_only` match `cglc_full` (then those
   components are not earning their keep).
5. The effect appears under `oracle_docs` but vanishes under `generic`
   (then the benefit depends on benchmark-supplied contract cues).

## Known limitations (stated up front)

* Worker and judge are the same model unless `--judge-model` is set, so
  self-judging bias is possible. Run a split-model replicate before claiming
  anything about judge quality.
* LLM calls are not perfectly reproducible even at temperature 0; report
  variation across seeds/replicates, not a single run.
* Small samples (tens of items) give wide CIs; the harness reports them
  rather than hiding them. Absolute F1 depends on the model and prompt, so
  compare arms, not papers.
* `arch3_fixed_checkpoints` approximates the historical design with this
  codebase's pieces (persistent ledger included).
* The corpus per item is small (<= ~10-20 paragraphs); this is not an
  open-corpus retrieval benchmark.
* Thresholds are the literature defaults, untuned (§14.7). Tuning happens on a
  dev split only, never on the reported items.

## Change log

(empty: no results have been produced under this protocol yet)
