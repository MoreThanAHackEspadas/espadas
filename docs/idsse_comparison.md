# IDSSE held-out comparison (rule: `docs/idsse_comparison_rule.md`)

**Named on 11 Nov 2026** by the team (Rochit), before any held-out IDSSE match (11, 14, 16, 17)
was scored, in line with rule 2:

Challenger: `357bcf2` `v4`

- Baseline (rule 1): `9ebf35d` `v4`, the submission on `main` (SHA-256 on our match `bc0a51d1…6291`).
- Challenger: `v4` at `357bcf2` on branch `refinement`: evidence models refitted and settings
  re-tuned on the 5-game development pool (Metrica 1 and 3 with the corrected stoppage labels,
  IDSSE 12, 13, 15), `X` off. Its build on our match: SHA-256 `7782b52f…56eb`, valid, deterministic
  (rule 4e, checked before naming).
- Why `v4` rather than `v3` (same commit): best leave-one-game-out score with re-tuning on the
  development pool, 0.5452 against 0.5408, and on the three IDSSE folds 0.543 against 0.532.
- Run by: Rochit, with `python tools/idsse_compare.py` from a clean checkout of the pushed commit
  that contains this file. The tool appends the results below.
