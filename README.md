![ARDC banner](demo/banner.png)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python 3.10+">
  <img src="https://img.shields.io/badge/tests-18_passing-brightgreen" alt="18 tests passing">
  <img src="https://img.shields.io/badge/architecture-frozen-orange" alt="architecture frozen">
  <img src="https://img.shields.io/badge/thresholds-tunable-yellow" alt="thresholds tunable">
  <img src="https://img.shields.io/badge/demo-live-orange" alt="live demo, no backend">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT license">
</p>

<h1 align="center">ARDC — Adaptive Research Depth Control</h1>
<p align="center"><b>The Contract-Gated Lease Controller (Worker, v1)</b><br>
An external control layer that stops document agents from quitting early,
rambling on, or digging in the wrong direction.</p>

<p align="center">
  <a href="#demo">Live demo</a> ·
  <a href="ARCHITECTURE.md">Architecture</a> ·
  <a href="PARAMETERS.md">Parameters</a> ·
  <a href="ROADMAP.md">Roadmap</a>
</p>

> **Status.** Working implementation of the frozen baseline
> (*Adaptive_Research_Depth_Final_Architecture_Corrected.docx*, 27 Aug 2026).
> Architecture — contract / trace / checkpoint semantics / effort ranking / work
> leases / finalization authorization — is fixed. Thresholds, weights and
> coefficients are tunable configuration with literature-backed defaults.

## Contents

- [Demo](#demo) · [Problem](#1-problem) · [Method](#2-method) ·
  [Parameters](#3-parameters) · [Repository map](#4-repository-map) ·
  [Reproduction](#5-reproduction) · [Evaluation](#6-evaluation-and-falsification) ·
  [Scope](#7-scope-and-limitations) · [Sources](#8-primary-sources) · [License](#license)

## Demo

![CGLC guided tour: stall detection, denied premature finish, and authorized full run](demo/demo-tour.gif)

*Figure 1 — Guided tour of the live demo (`demo/index.html`): repeated queries trigger the
structural-stall checkpoint; a fluent but unsupported draft is denied finalization; a full
run terminates in `ALLOW_FINALIZE` with a complete audit record. Recorded in Chromium
(19 s, 1.3 MB); full-quality video: [`demo/demo-tour.mp4`](demo/demo-tour.mp4).*

Run it yourself — no backend, the full controller executes in the page:

- **Interactive explainer + console:** open `demo/index.html` (beginner-friendly:
  problem → idea → 5 steps → live console → parameters → falsification), press
  **Guided tour**, drag the sliders to change policy behavior.
- **Terminal version:** `python examples/demo_run.py` (stall trace, gate block, full
  audit trail).
- **Hosted:** `docs/index.html` is Pages-ready (repo Settings → Pages → `main`/`docs`).

![CGLC console after a guided tour: run log, lease timeline, audit record](demo/screenshot.png)

## 1. Problem

Adaptive Research Depth addresses *effort miscalibration* in document-grounded work:

| # | Failure mode | Observable consequence |
|---|--------------|------------------------|
| 1 | Under-execution | Plausible answer before required reading, comparison, or verification |
| 2 | Unproductive continuation | Repeated actions / low-yield retrieval after sufficient work |
| 3 | Misdirected effort | Further budget in the same direction cannot close the relevant gap |

Fixed budgets, token caps, and generic "be thorough" prompts cannot distinguish
*completion* from *local stagnation* from *blocked termination*. CGLC makes that
distinction structural: the worker may propose completion but can never self-authorize it.

## 2. Method

**Runtime loop.** The worker executes inside a lease while the guard plane logs raw events
and costs. A semantic checkpoint fires only on mandatory events — finalization request,
lease expiry, `p`-round structural stall, blocker/contradiction, budget-tier crossing, or
maximum silence. At the checkpoint, the controller assembles the minimal semantic packet
`S_t = Φ(K, Δτ_t, L_{t−1}, draft_t, ℓ_t, budget_t)`, obtains a structured
receipt-cited judgment, and applies the deterministic decision order of §14.6:

1. Conjunctive gate passes → `ALLOW_FINALIZE`
2. User-only information needed → `ASK_USER`
3. Infeasible under access/budget → `REPORT_BLOCKED`
4. Otherwise rank feasible non-terminal actions by value-per-cost and issue the next lease;
   on non-positive utility fall back to `VERIFY` → `REDIRECT` → shortest safe lease.

![CGLC runtime loop: contract, worker, guards, checkpoint, policy, and terminal outcomes](docs/architecture.svg)

**Normative equations (§14).** Retrieval novelty `UPR_t = |C_t ∖ H_{t−1}| / max(1, |C_t|)`;
action similarity `J_t` as windowed Jaccard maxima; `Stagnated_t` over `p` consecutive
rounds — a lease-ending inspection trigger, never a completion signal. Budget pressure
`ρ_t = 1 − min_j clip(b_jt/B_j, 0, 1)`; normalized cost `d_t(a)`; pressure
`Π_t(a) = λ_b·ρ_t·d_t(a)`. Operational score `u_t(a) = Δ̂_t(a) + Ψ_t(a) − Π_t(a)`,
`r_t(a) = max(0, u_t)/(d_t + ε)` — a ranking proxy among allowed non-terminal actions,
not a value-of-information claim and incapable of overriding any gate:

```
ALLOW_t = C_terminal ∧ C_process ∧ C_evidence ∧ C_answer ∧ ¬C_blocker
```

Exhaustion without a passing gate yields `REPORT_BLOCKED` (a transparent partial report),
never a labeled success. Every checkpoint emits a compact audit record (contract/lease
versions, trigger, receipt IDs, gate snapshot, selected and rejected decisions, remaining
budget, controller cost). Full design rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 3. Parameters

Defaults are extracted from their originating papers — starting points, not optima —
and must be ablated on held-out trajectories. Full provenance in [`PARAMETERS.md`](PARAMETERS.md);
runnable values in [`configs/default.yaml`](configs/default.yaml).

| Component | Symbols | Default | Origin |
|---|---|---|---|
| Stagnation gate | `τ_J, τ_U, p` | `0.6, 0.3, 2` | CGDP (`f_j0.6_u0.3_p2`) |
| Action utility | `λ_b, λ_decomp, λ_early` | `0.7, 0.14, 0.18` | Inference-Time Budget Control |
| Resource weights / smoothing | `ω_j, ε` | equal, `1e−5` | same |
| HALT stopping | `τ, ε, policy` | `0.0, 0.02, ALL_MATCH` | HALT |
| Coverage (ablation) | `θ_e, θ_ev, δ, r_max` | `0.85, 0.70, 0.05, 7` | MAP-Law |
| Continuation (deferred) | `λ decay, T` | `1.0→0.1, 10` | Stop-RAG |
| Observation filter | `τ_len, τ_step, τ_loop` | `3000, 4–8, 3–5` | SupervisorAgent |
| Execution temperature | `T_gen, T_select` | `0.7, 0.0` | BATS |
| Leases | SHORT / STANDARD / EXTENDED | `1 / 3 / 5` | project convention |

## 4. Repository map

```
src/cglc/
  config.py        all parameter groups + budget tiers (literature defaults)
  contracts.py     task contract K (goal, duties, obligations, schema, blockers)
  trace.py         append-only raw trace + budget accounting
  ledger.py        bounded evidence ledger (support s_it, contradiction c_it)
  stagnation.py    UPR_t, J_t, Stagnated_t
  budget.py        rho_t, d_t(a), Pi_t(a)
  scoring.py       Delta_hat, Psi, u_t, r_t
  gates.py         conjunctive finalization gate (+ HALT / coverage helpers)
  leases.py        SHORT/STANDARD/EXTENDED + guard-plane screens
  guards.py        deterministic guards + checkpoint scheduler
  controller.py    decision order (§14.6) + fallback
  runner.py        event-driven Fig. 2 loop (pluggable LLM judge)
  audit.py         decision records (§7.5)
  worker/          DocumentWorker interface, FixedCorpusWorker, WorkerAdapter
  evaluate/        Table-16 metrics + §11.1 ablation ladder
demo/              banner, explainer + live demo, GIF, MP4, screenshots
docs/              Pages-ready demo copy + architecture figure
configs/           default.yaml (runnable literature defaults)
examples/          quickstart.py, comparison_task.json, demo_run.py
tests/             18 unit/integration tests
```

## 5. Reproduction

```bash
pip install -e ".[dev]"
pytest -q                        # 18 tests: stagnation, budget, scoring, gates, controller, runner
python examples/quickstart.py    # minimal controlled comparison run
python examples/demo_run.py      # narrated terminal demo: stall, gate block, full run w/ audit
```

To substitute a real checkpoint judge, implement `JudgeFn(contract, ledger, draft)` per
`runner.py` (structured, low-variance call citing receipt IDs; keep the
`LOW/MEDIUM/HIGH → 0/0.5/1` mapping fixed before evaluation).

## 6. Evaluation and falsification

Minimum ladder (§11.1): uncontrolled worker → fixed-shallow / fixed-deep → generic prompt →
fixed checkpoints → CGLC∖stall → CGLC∖scoring → **full CGLC** (+ optional always-on judge
as overhead upper bound). Dimensions (Table 16): outcome quality, process compliance,
grounding, behavior, user time, system cost, controller errors, calibration. The project
fails if a prompt/fixed budget matches it, overhead consumes savings, premature answers
rise, redirects never beat same-direction budget, gains vanish off-distribution, or the
ledger duplicates worker effort. What to build next, in order: [`ROADMAP.md`](ROADMAP.md).

## 7. Scope and limitations

Version 1 covers fixed-corpus comparison, document QA, and specification audit with
benchmark/user-confirmed contracts and no training. Deferred: automatic obligation
extraction, learned continuation values and lease sizing, calibrated false-allow control,
open-web scope, and controller-generated exact plans (explicitly out of scope —
that would make the controller a second research agent). No claim is made of globally
optimal stopping or global evidence exhaustion; gates are contract-relative, and stagnation
signals local stall only.

## 8. Primary sources

WebRider (intent contract, ledger, gated success) · CGDP/PBAI (bounded memory, stagnation
trigger) · Inference-Time Budget Control (feasible ranking, value-per-cost) · HALT
(external evidence-relative stopping) · Stop-RAG (forward continuation value) · MAP-Law
(requirements, gain, direction) · SupervisorAgent (cheap screening) · ECT (authorization
vs proposal) · BATS (budget tracking, continue/pivot) · S2G-RAG (upper-bound judge) ·
RaM (value-of-computation objective). Paper-specific figures cited are author-reported;
the synthesis remains a hypothesis pending the evaluation above.

## License

MIT — see [LICENSE](LICENSE).
