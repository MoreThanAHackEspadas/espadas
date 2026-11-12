# Refinement 2: adoption rule (branch `refinement-2`)

Written and pushed on **11 Nov 2026**, before any leave-one-game-out result of this branch was
read. It decides whether anything from `refinement-2` replaces the submission on `main`. Not
edited after the first result; results go to `docs/refinement2_result.md`.

1. **Frozen submission:** `main@53861c0`, `v4`, `X` off, SHA-256 on our match `7782b52f…56eb`.
   It ships unless a candidate is adopted under this rule.
2. **Why leave-one-game-out (LOGO).** No untouched external game is left: Metrica Game 2 and
   IDSSE 11, 14, 16, 17 have each had their one comparison. LOGO (`tools/logo.py`): for every
   game of a pool, the evidence models are re-fitted and the settings re-tuned on the other
   games, and the game left out is scored with them.
3. **Score:** spellF1 on the left-out game, mean over the four tracker levels at 25% live ball
   hidden, `X` off; then the mean over the folds. Robustness: the same fold models and settings
   scored on the left-out game at 15% and 35% hidden (not re-tuned for those).
4. **Baseline:** the frozen method (the code of `main@53861c0`: five-match pool, its search
   grid, logistic evidence models), run through LOGO by this branch's `tools/logo.py`, which
   differs from `main`'s only by the robustness scores.
5. **Adopt a candidate only if all of:**
   (a) its LOGO mean beats the baseline's by **at least 0.02**;
   (b) it is ahead of the baseline on **at least 70% of the games compared** (4 of 5, 7 of 10);
   (c) it is **not behind** the baseline at 15% or at 35% hidden (mean over the games);
   (d) its build on our match passes the validator and `tools/check_determinism.py`.
   Otherwise the frozen submission stays. Ties stay.
6. **Pools of different size (step 2, ten matches).** Both methods are scored on all ten
   external games, each game by models that never saw it: the candidate by ten-fold LOGO on the
   full pool; the baseline by five-fold LOGO on its own five games, and by its shipped models and
   settings (`main@53861c0`) on the five games it was never fitted on (Metrica 2; IDSSE 11, 14,
   16, 17). Rule 5 applies to those ten per-game scores.
7. **Steps are judged in sequence.** Each step is measured against the step before it, to know
   what helped, but adoption is decided once, for **one final candidate named by the end of
   14 Nov 2026**, against the baseline of rule 4 / 6. A step that does not improve LOGO over
   the previous step is dropped before the next one is tried.
8. **Not used to decide:** any number from our match (possession count, dead share), and the
   in-sample (training) scores.
9. **Known weaknesses, accepted:** LOGO's descent starts from the shipped settings, which were
   tuned on all five pool games, the same for baseline and candidate; the folds share games, so
   they are not independent; and choosing among several steps by LOGO is itself a form of
   tuning on it. That is why (a) asks for 0.02 and (b) for consistency across games, and why the
   number of steps tried is reported in the result file.
10. **If adopted:** merge into `main`, new SHA-256 in the README and tech doc, tech doc updated,
    fresh-clone rebuild, determinism, data guard, then the tag. **If not:** `main@53861c0` is
    tagged and submitted.
