# Architecture Mapping (frozen baseline 27 Aug 2026)

Source: `Adaptive_Research_Depth_Final_Architecture_Corrected.docx`.
Normative math: Sec 14. Where shorthand elsewhere conflicts with Sec 14,
Sec 14 wins.

## Decision (Sec 1)

CGLC: external controller around a capable document worker. Stable task
contract, cheap trace, bounded leases, event-driven semantic checkpoints,
conjunctive finalization gate. Asymmetric: finalization always
intercepted; over-execution screened cheaply; rich judgment paid at lease
boundaries / consequential events.

## Planes -> code (Sec 3.2)

| Plane | Responsibility | Module |
|---|---|---|
| Contract | `K` policy; changes only on clarification/revision | `contracts.py` |
| Execution | worker acts within lease; raw trace logged | `worker/`, `trace.py` |
| Guard | budgets, finalization intercept, stall/blocker detect | `guards.py`, `leases.py`, `stagnation.py`, `budget.py` |
| Checkpoint | on-demand `S_t = Phi(K, delta-tau, L, draft, lease, budget)` | `runner.py` |
| Policy | finalize vs next bounded lease; `r_t` ranks non-terminal only | `controller.py`, `scoring.py` |
| Evaluation | offline quality/compliance/grounding/time/cost/errors | `evaluate/` |

## State (Sec 4)

- `K = (g, H_proc, H_evid, S_soft, B_mat, A_schema, B_policy)`
- `tau_t` continuous raw trace; `L_t` checkpoint-persisted ledger;
  `ell_t = (intent, target_gaps, allowed_classes, cap, expected_test)`;
  `S_t` built on demand only.
- Ledger per-obligation bounded (strongest + <=2 corroborating +
  contradictions); artifacts stay in trace store.

## Runtime (Sec 5 + 14)

1. Scheduler fires on finalize / lease-expiry / `p`-round stall /
   blocker / contradiction / budget-tier / `L_max` silence.
2. `J_t`, `UPR_t`, `Stagnated_t` per Sec 14.2 (never finalizes).
3. Semantic judgment returns sufficiency / gaps / verification need /
   effectiveness / direction / alternative / uncertainty (receipt-cited).
4. `A_feas` filtered by `deterministic_guards`; `u_t`, `r_t` per 14.4-14.5.
5. Decision order per 14.6: gate -> ASK -> BLOCKED -> rank -> lease;
   fallback VERIFY > REDIRECT > shortest lease > ASK/BLOCKED.
6. Leases SHORT/STANDARD/EXTENDED = 1/3/5; stall/blocker/contradiction
   ends any lease early.

## Guarantees (Sec 7)

`ALLOW = C_terminal & C_process & C_evidence & C_answer & !C_blocker`.
Exhaustion -> `REPORT_BLOCKED` (partial answer allowed, never labeled
success). Failure-safe defaults: uncertain decisive claim -> VERIFY;
low-yield + alternative -> REDIRECT; no safe action -> ASK/BLOCKED;
inconsistent contract -> stop + revise. Every checkpoint audited
(`audit.py`).

## Literature use (Sec 8/13)

WebRider: contract/ledger/guarded finalization kept; controller-owned
Action-AST rejected. CGDP: bounded memory + stagnation trigger kept;
per-observation extractor + force-finalize rejected (renamed structural
stagnation). Budget Control: feasible ranking + value-per-cost kept as
experimental policy with logged terms. HALT/Stop-RAG/MAP-Law/Supervisor/
ECT/BATS/S2G/RaM contribute bounded mechanisms per Table 13.

## Build order (Sec 10.1)

Contract schema -> worker adapter + interceptor -> guards/budget/stall
trigger -> checkpoint + structured judgment -> finalization gate ->
leases/CONTINUE-VERIFY-REDIRECT -> logged scorer -> offline eval before
auto-extraction / learned values / open web.

## Falsification (Sec 11.3)

Kill CGLC if: prompt/fixed budget matches it; overhead eats savings;
premature/unsupported answers rise; REDIRECT never beats same-direction
budget; gains vanish off-distribution; ledger duplicates worker effort.
