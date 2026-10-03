# ARDC Roadmap — how to improve from here

Ordered by evidence value per effort. Items move top-to-bottom; nothing below
starts until the item above has measurements.

## Now (no new architecture)

1. **Real benchmark harness.** *(Harness built, protocol frozen in BENCHMARK.md; no real results yet.)* Wrap HotpotQA / 2WikiMultiHopQA / MuSiQue items as
   `TaskContract`s (gold supporting facts → `H_evid`, question type → `H_proc`),
   run the §11.1 ladder, publish the results table. This is the single highest-value
   step: it turns literature-backed defaults into *measured* claims.
2. **LLM checkpoint judge.** Implement `JudgeFn` with a structured low-temperature
   call (receipt-cited JSON per `runner.py`), keeping `LOW/MEDIUM/HIGH → 0/0.5/1`
   frozen. Ablate rule-judge vs LLM-judge on controller errors (false allow/block,
   wrong redirect).
3. **Calibration sweeps.** Grid `τ_J × τ_U × p` and `λ_b × λ_v,r,g,l` on a dev split;
   report quality/cost curves so no result depends on one favorable setting (§14.7).
4. **CI + hygiene.** GitHub Actions (`pytest -q` on push), `ruff`, coverage gate,
   `CITATION.cff`, changelog, versioned releases. Converts "works on my machine"
   into a verifiable project.

## Next (measured extensions, §10.3 order)

5. **Automatic obligation extraction** as a separately scored component (contract
   provenance logged; never silently mixed with benchmark contracts).
6. **Learned continuation value** (Stop-RAG style Q(λ), `1.0→0.1` cosine, `T=10`)
   as an ablation against the transparent scorer — not a replacement until it wins.
7. **Calibrated false-allow control** on labeled trajectories (HALT-style
   non-inferiority margin `ε=0.02` as the starting operating point).
8. **Redirect quality study.** Does `REDIRECT` actually recover better than
   same-direction budget? If not, the direction machinery must be redesigned.

## Later (scope changes — explicit decisions, not drift)

9. Open-web corpus with changing source access and unbounded query space.
10. Multi-worker / multi-lease scheduling under a global budget.
11. Controller-generated exact redirect plans (redefines the project as research
    orchestration; requires a new architecture record).

## Non-goals (stay out unless the architecture record is rewritten)

- Global optimality or exhaustion claims; per-instance truth certificates.
- Controller-owned full research plans (violates the externality premise, §3.3).
- Tuning defaults on test distributions (§14.7 violation).
