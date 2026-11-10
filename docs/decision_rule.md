# Selection decision rule (checklist 9.3)

Adopted by the team on **10 Nov 2026**, written and pushed **before** the one-shot held-out
comparison (`tools/selection.py`). Game 2 has not been scored on the current bench at the time of
writing. The result is applied as it comes out; this file is not edited after the run.

1. **Pre-registered choice: `v4`.** Best mean spellF1 on the tuning games (Metrica 1 + 3, four
   tracker levels, 25% live ball hidden): `v4` 0.555, `v3` 0.539, `v1` 0.484. `v4` is ahead at
   every tracker level; its lead over `v1` (0.071) is outside the organisers' 0.04 spellF1 band.
2. **Main score:** Game 2 spellF1, mean over the four tracker levels at 25% live ball hidden,
   `X` off.
3. **Switch away from `v4` only if** another candidate
   (a) beats it on the main score by **at least 0.04** (the organisers' spellF1 guidance,
   `possession_metrics.SPELL_NOISE_BAND`), **and**
   (b) leads it at **3 or more of the 4** tracker levels, **and**
   (c) is **not behind** `v4` at 15% or at 35% live ball hidden.
   If two candidates qualify, take the one with the bigger margin.
4. **Anything inside the band is a tie, and a tie keeps `v4`.** macroF1 is never used to pick the
   variant. The tuning-versus-Game-2 drop is reported, not used.
5. **`X` stays off** whatever its macroF1 gain: on `v4`, 89% of its frames are Metrica's ball-less
   stoppages, the pattern decision (b) refused to rely on (`espadas/abstain.py`).
6. **The provided match plays no part.** Its aggregate counts (possessions per 90 minutes, dead
   share) are never used to choose. If the winner fails a hard check (validator, determinism),
   the next candidate under rule 3 is used; the comparison is not re-run.
7. **Order:** this rule is pushed first; `tools/selection.py` is then run once, by one named person,
   from the main checkout at the pushed commit, with no extra flags; `docs/selection.md` is
   committed and pushed straight after, with the Decision line filled in from this rule.

Why a pre-registered default with a high bar: the tuning scores are optimistic (each labeler's
settings were chosen on them, `v4` has the most settings), while Game 2 is unbiased but a single
match, where differences of a few hundredths are mostly chance. The default carries the tuning
evidence; Game 2 can overturn it only by a large, consistent margin. The costly mistake is
`v4` against `v1` (about 0.07 on the tuning games), which this rule still catches; a wrong call
between `v4` and `v3` costs about 0.01–0.02.
