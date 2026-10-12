# Clerk eval results (offline)

> **Offline simulator run.** These numbers check that the pipeline is wired correctly. They are NOT a measure of Nemotron's quality. Run with `CLERK_MODE=live` for real numbers.

- Cases: **50** synthetic statements, **80** planted errors worth **$948.82**, plus 43 decoys (legitimate charges that look wrong)
- Routing: Nano for reading/auditing, Super for judging/writing
- Model calls: 293 · tokens: 162,523 in / 20,948 out · cost: not configured (set CLERK_PRICE_* env vars)
- Average time per bill: 0.00s
- Failures: 0

| Stage | Precision | Recall | F1 | Correct $ found | $ wrongly accused |
|---|---|---|---|---|---|
| Auditors only | 54% | 100% | 0.70 | $948.82 | $1,713.33 |
| + Verifier (code) | 56% | 100% | 0.72 | $948.82 | $1,588.33 |
| + Skeptic (what users see) | 100% | 100% | 1.00 | $948.82 | $0.00 |

_Generated 2026-10-12 06:09 UTC by `python -m evals.run_eval`._
