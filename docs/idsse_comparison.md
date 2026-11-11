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

## Run started 2026-11-11 15:06 (tool at `5f76c8f`; baseline `9ebf35d`, challenger `357bcf2` v4)

**25% live ball hidden** (spellF1; possessions predicted / reference)

| Candidate | Match | light | medium | heavy | smooth | Mean |
|---|---|---|---|---|---|---|
| base | 11 | 0.490 (257/395) | 0.462 (265/395) | 0.434 (278/395) | 0.442 (259/395) | 0.4570 |
| base | 14 | 0.447 (283/400) | 0.427 (290/400) | 0.390 (297/400) | 0.418 (295/400) | 0.4205 |
| base | 16 | 0.445 (278/359) | 0.452 (283/359) | 0.391 (278/359) | 0.404 (269/359) | 0.4228 |
| base | 17 | 0.472 (249/319) | 0.456 (248/319) | 0.406 (242/319) | 0.427 (237/319) | 0.4405 |
| **base** | **mean** | 0.4634 | 0.4495 | 0.4053 | 0.4228 | **0.4352** |
| chal | 11 | 0.586 (276/395) | 0.555 (278/395) | 0.529 (286/395) | 0.547 (287/395) | 0.5541 |
| chal | 14 | 0.565 (287/400) | 0.571 (300/400) | 0.535 (325/400) | 0.538 (296/400) | 0.5523 |
| chal | 16 | 0.566 (286/359) | 0.566 (306/359) | 0.527 (294/359) | 0.523 (294/359) | 0.5455 |
| chal | 17 | 0.562 (258/319) | 0.587 (251/319) | 0.564 (257/319) | 0.550 (257/319) | 0.5657 |
| **chal** | **mean** | 0.5699 | 0.5698 | 0.5385 | 0.5394 | **0.5544** |

**15% live ball hidden** (spellF1; possessions predicted / reference)

| Candidate | Match | light | medium | heavy | smooth | Mean |
|---|---|---|---|---|---|---|
| base | 11 | 0.500 (274/395) | 0.483 (274/395) | 0.451 (294/395) | 0.443 (261/395) | 0.4692 |
| base | 14 | 0.455 (292/400) | 0.431 (298/400) | 0.401 (309/400) | 0.424 (300/400) | 0.4277 |
| base | 16 | 0.449 (276/359) | 0.458 (286/359) | 0.383 (285/359) | 0.424 (282/359) | 0.4286 |
| base | 17 | 0.496 (256/319) | 0.468 (251/319) | 0.413 (250/319) | 0.446 (238/319) | 0.4558 |
| **base** | **mean** | 0.4750 | 0.4601 | 0.4121 | 0.4342 | **0.4453** |
| chal | 11 | 0.599 (281/395) | 0.584 (286/395) | 0.550 (293/395) | 0.547 (293/395) | 0.5701 |
| chal | 14 | 0.590 (296/400) | 0.586 (313/400) | 0.544 (323/400) | 0.551 (300/400) | 0.5681 |
| chal | 16 | 0.572 (290/359) | 0.575 (310/359) | 0.531 (300/359) | 0.533 (298/359) | 0.5529 |
| chal | 17 | 0.595 (267/319) | 0.597 (259/319) | 0.572 (261/319) | 0.562 (252/319) | 0.5815 |
| **chal** | **mean** | 0.5890 | 0.5858 | 0.5495 | 0.5483 | **0.5681** |

**35% live ball hidden** (spellF1; possessions predicted / reference)

| Candidate | Match | light | medium | heavy | smooth | Mean |
|---|---|---|---|---|---|---|
| base | 11 | 0.466 (254/395) | 0.447 (263/395) | 0.416 (265/395) | 0.418 (250/395) | 0.4369 |
| base | 14 | 0.431 (272/400) | 0.394 (279/400) | 0.380 (281/400) | 0.394 (280/400) | 0.3997 |
| base | 16 | 0.432 (265/359) | 0.433 (275/359) | 0.373 (269/359) | 0.392 (262/359) | 0.4073 |
| base | 17 | 0.450 (240/319) | 0.432 (238/319) | 0.404 (244/319) | 0.417 (236/319) | 0.4260 |
| **base** | **mean** | 0.4448 | 0.4267 | 0.3933 | 0.4051 | **0.4175** |
| chal | 11 | 0.555 (285/395) | 0.537 (272/395) | 0.501 (281/395) | 0.524 (281/395) | 0.5292 |
| chal | 14 | 0.552 (286/400) | 0.519 (284/400) | 0.507 (311/400) | 0.530 (294/400) | 0.5270 |
| chal | 16 | 0.537 (279/359) | 0.536 (288/359) | 0.530 (294/359) | 0.506 (289/359) | 0.5273 |
| chal | 17 | 0.556 (256/319) | 0.566 (248/319) | 0.561 (252/319) | 0.518 (254/319) | 0.5502 |
| **chal** | **mean** | 0.5499 | 0.5395 | 0.5249 | 0.5195 | **0.5334** |

**Rule 4** (base = `9ebf35d` v4, chal = `357bcf2` v4):
- (a) pass: margin +0.1192 (needs >= +0.04)
- (b) pass: ahead on 4 of 4 matches (needs 3)
- (c) pass: ahead at 4 of 4 tracker levels (needs 3)
- (d) pass: at 15% / 35% hidden: +0.1228 / +0.1160
- (e) validator + determinism of the challenger's build on our match: checked separately, before merging

**Result of (a)-(d): challenger qualifies (subject to (e))**

**Decision:** **adopt the challenger, `357bcf2` `v4`, `X` off** (rule 4 and 7). (a) +0.1192
against the 0.04 bar; (b) ahead on all 4 held-out matches (+0.097 to +0.130); (c) ahead at all 4
tracker levels; (d) ahead at 15% (+0.123) and 35% (+0.116); (e) its build on our match was checked
before naming: valid, deterministic, SHA-256 `7782b52f4b58419f0289e273ee0ae93faa6e4815c949ad12f61a887ad8e456eb`.
Next (rule 7): merge into `main`, update the README and tech doc SHA-256, then the freeze on the
merged commit before 16 Nov. The out-of-sample estimate beforehand was about +0.10
(leave-one-game-out); the held-out margin, +0.12, is in line with it.
