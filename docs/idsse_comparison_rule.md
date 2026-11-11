# IDSSE held-out comparison: decision rule (branch `refinement`)

Written and pushed on **11 Nov 2026**, before any held-out IDSSE match (11, 14, 16, 17) was
scored by anyone, with any labeler. It decides whether anything from `refinement` replaces the
submission on `main`. The result is applied as it comes out; this file is not edited after the
run.

1. **Baseline: the submission as frozen on `main` at `9ebf35d`**: `v4`, `X` off, with that
   commit's code, evidence models (`espadas/evidence_models.py`) and settings. Its file on our
   match has SHA-256 `bc0a51d1…6291`. The baseline is never re-fitted or re-tuned.
2. **One challenger.** By the end of **14 Nov 2026** the team names one challenger: a commit on
   `refinement` and a labeler (with `X` off). Anything may be tried before that on the tuning
   data (Metrica 1 + 3, IDSSE 12, 13, 15); only the named challenger is scored on the held-out
   matches. No challenger named by then means no comparison and `main` ships as it is.
3. **Main score:** spellF1, mean over the **four held-out IDSSE matches × the four tracker
   levels** (light, medium, heavy, smooth), with the ball hidden on 25% of live frames, `X` off,
   against the IDSSE reference of `espadas/external/idsse.py` as of `48a5a8c` (provider flags plus
   the fixed 1 s touch rule). Same degradation code for both candidates.
4. **Adopt the challenger only if all of:**
   (a) it beats the baseline on the main score by **at least 0.04** (the organisers' spellF1
   guidance, `possession_metrics.SPELL_NOISE_BAND`);
   (b) it is ahead on **at least 3 of the 4** held-out matches (mean over the levels);
   (c) it is ahead at **at least 3 of the 4** tracker levels (mean over the matches);
   (d) it is **not behind** at 15% or at 35% live ball hidden (mean over matches and levels);
   (e) its build on our match passes the validator and `tools/check_determinism.py`.
   Otherwise the baseline stays. A tie stays. macroF1 is not used.
5. **Not used to decide:** Metrica Game 2 (spent at selection, 10 Nov), any number from our
   match (possession counts, dead share), and the tuning matches' scores. They may be reported.
6. **Order:** the comparison tool is written and pushed first, and checked on the tuning matches
   only; the challenger is named in writing (commit + labeler) in `docs/idsse_comparison.md`;
   then the tool is run **once**, by one named person, from a clean checkout of the pushed
   commit; its output is committed straight after, with the Decision line filled in from this
   rule. A run that crashes before scoring may be repeated; a completed run may not.
7. **If adopted:** the challenger is merged into `main`, becomes `DEFAULT_LABELER` (or its
   settings replace `v4`'s), the README and tech doc SHA-256 are updated, and the freeze (tag,
   clean-clone rebuild, determinism, data guard) is done on the merged commit before 16 Nov.
   **If not:** `main` ships at `9ebf35d` unchanged, and `refinement` stays a branch.

Why this bar: four new matches from a different league and tracker are a much better test than
one Metrica game, but the challenger will have been developed with the tuning matches in view
and the cost of a wrong switch two days before the deadline (a new, less-checked build) is
higher than the cost of missing a small gain. 0.04 is the organisers' own noise guidance; the
per-match and per-level counts make sure the gain is broad, not one match or one tracker level.
