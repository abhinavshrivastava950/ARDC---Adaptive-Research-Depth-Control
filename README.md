# CGLC Worker — Contract-Gated Lease Controller (V1)

**Live demo:** open `demo/index.html` in a browser (or `docs/index.html` via
GitHub Pages once enabled in repo Settings → Pages → Deploy from `main`/`docs`).
No backend — the full run executes in the page.

**Video tour (20 s):** [`demo/demo-tour.mp4`](demo/demo-tour.mp4) — scrolls the
story, runs the guided tour (stall → denied finish → full run) in a real browser.

![CGLC live demo — full comparison run](demo/screenshot.png)

Architecture baseline: `Adaptive_Research_Depth_Final_Architecture_Corrected.docx`
(frozen 27 Aug 2026). This repo implements the **worker + external controller
loop**; thresholds stay tunable. See `ARCHITECTURE.md` (design mapping),
`PARAMETERS.md` (literature defaults), `configs/default.yaml` (runnable config).

## Rule

Hard obligations constrain what is allowed. Semantic sufficiency governs
whether the answer is ready. Stagnation and value-per-cost govern what to
do next. Never collapsed into one score.

## Layout

- `src/cglc/config.py` — all 7 literature parameter groups + lease/budget tiers
- `src/cglc/contracts.py` — `K = (g, H_proc, H_evid, S_soft, B_mat, A_schema, B_policy)`
- `src/cglc/trace.py` — append-only `tau_t` + budget accounting
- `src/cglc/ledger.py` — bounded evidence ledger (`s_it`, `c_it`, receipts)
- `src/cglc/stagnation.py` — `UPR_t`, `J_t`, `Stagnated_t` (CGDP `f_j0.6_u0.3_p2`)
- `src/cglc/budget.py` — `rho_t`, `d_t(a)`, `Pi_t(a)`
- `src/cglc/scoring.py` — `Delta_hat`, `Psi`, `u_t`, `r_t` (ranking proxy only)
- `src/cglc/gates.py` — `ALLOW = C_terminal & C_process & C_evidence & C_answer & ~C_blocker`
- `src/cglc/leases.py` — SHORT=1 / STANDARD=3 / EXTENDED=5 + guard screens
- `src/cglc/guards.py` — deterministic guards + checkpoint scheduler
- `src/cglc/controller.py` — decision order (Sec 14.6) + fallback
- `src/cglc/runner.py` — Fig. 2 event-driven loop (plug in an LLM `judge`)
- `src/cglc/worker/` — `DocumentWorker` interface, `FixedCorpusWorker`, `WorkerAdapter`
- `src/cglc/evaluate/` — Table 16 metrics + Sec 11.1 ablation list
- `examples/quickstart.py` — fixed-corpus comparison demo

## Quickstart

```bash
pip install -e ".[dev]"
python examples/quickstart.py
pytest -q
```

## V1 scope (Sec 10.2)

Fixed-corpus comparison / QA / spec audit. Benchmark/user-confirmed
contracts. Structured checkpoint judgments. `1/3/5` leases. No training.

## Adding an LLM judge

Implement `JudgeFn(contract, ledger, draft) -> Judgment` in `runner.py`
(e.g. low-temperature structured call citing receipt IDs) and pass it to
`Runner(judge=...)`. Keep `LOW/MEDIUM/HIGH -> 0/0.5/1` fixed before eval.
