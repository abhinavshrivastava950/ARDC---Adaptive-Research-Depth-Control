# CGLC Parameter Reference (literature-backed defaults)

> Starting points, not universal optima. Log every value in the run record;
> ablate on held-out trajectories (Sec 14.7).

| Component | Symbol | Default | Origin | Validation |
|---|---|---|---|---|
| Stagnation gate | `tau_J, tau_U, p` | `0.6, 0.3, 2` | CGDP (`f_j0.6_u0.3_p2`) | LoCoMo, MuSiQue, SWE-QA-Pro x ReAct/IRCoT/MemGPT/Iter-RetGen; -39% tokens, +1.1/+2.8/+6.9 acc |
| Action utility | `lambda_b, lambda_decomp, lambda_early_ans` | `0.7, 0.14, 0.18` | Inference-Time Budget Control | 48 cells (HotpotQA/2Wiki/MuSiQue/Bamboogle x Low-High budgets x Qwen3-32B/GPT-5.4-Mini/Qwen3.5-122B); -27.2% wall-clock |
| Resource weights | `omega_j` | equal, sum `1.0` | same | same |
| Smoothing | `epsilon` | `1e-5` | same | same |
| HALT stopping | `tau, epsilon, policy` | `0.0, 0.02, ALL_MATCH` | HALT | HotpotQA/2Wiki/MuSiQue, Self-Ask 3B/7B; -20..-45% loops |
| Graph coverage | `theta_e, theta_ev, delta` | `0.85, 0.70, 0.05` | MAP-Law | 30 labor-law pilots; coverage 1.000, rounds 7.0->3.36 |
| Max rounds | `r_max` | `7` | MAP-Law | same |
| Stop-RAG | `lambda decay, T` | `1.0->0.1 cosine, 10` | Stop-RAG | Llama-3.1-8B + DeBERTa-v3-large; +2.2..2.6 F1 |
| Observation filter | `tau_len, tau_step, tau_loop` | `3000, 4-8, 3-5` | SupervisorAgent | GAIA/HumanEval/MBPP/GSM-Hard/AIME/DROP; GAIA 30.0->46.7% |
| Execution temp | `T_gen, T_select` | `0.7, 0.0` | BATS | BrowseComp/ZH, HLE-Search, tau2-bench; flat ~14-16% |
| V1 Psi init | `lambda_v, lambda_r, lambda_g, lambda_l` | `0.18, 0.14, 0.14, 0.10` | transparent init reusing origin magnitudes | **must be calibrated**; architecture fixes form, not values |
| Delta map | `LOW/MEDIUM/HIGH` | `0 / 0.5 / 1` | V1 convention | fix before test |
| Leases | `SHORT/STANDARD/EXTENDED` | `1 / 3 / 5` | Sec 5.6/10.2 | logged + tunable |
| Ledger support | `s_it` | `{0, 0.5, 1}` = UNSEEN/PARTIAL/SUPPORTED | Sec 14.3 | — |
| Contradiction | `c_it` | `{0,1}` | Sec 14.3 | separate from support |
