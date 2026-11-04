"""Fit the three evidence models (checklist 5.10) on the external TUNING games only.

    python tools/fit_evidence.py              # fits, prints a report, writes espadas/evidence_models.py
    python tools/fit_evidence.py --heldout    # also reports the held-out game (opt-in, checklist 4.10)

Training rows: every graded frame of Metrica Games 1 and 3, once per ball blur level (light,
medium, heavy; degrade.BLUR), with the ball hidden on degrade.LIVE_MISSING of live frames (the
same degraded views the labelers are tuned on).
- dead model, in two parts (see evidence.DEAD_SPLIT), both reference D vs H/A with an intercept:
  "motion" (players only) fitted on all graded frames, "ball" (players + ball) on the graded
  frames where the ball is tracked.
- team model: reference H vs A on live frames only, no intercept (every team clue is
  home-minus-away, so the model is symmetric between the teams).
- control model: reference "ball at a player's feet" (events.control_mask) vs loose, on live
  frames where the ball is tracked; with an intercept.
Labels come from the games' events by code (events.py). The held-out game (Metrica 2) is only
reported, never fitted on, and only when --heldout is given. Our match is never touched. Deterministic.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config, evidence  # noqa: E402
from espadas.external import bench, degrade, events  # noqa: E402

OUT = os.path.join(ROOT, "espadas", "evidence_models.py")
HIDE_BALL = degrade.LIVE_MISSING
RIDGE = 1.0


BLURS = ("light", "medium", "heavy")   # fit on every blur level (checklist 6.6), as tune.py


def rows(numbers, hide=HIDE_BALL, blurs=BLURS):
    """[(name, reference labels, clues, graded mask, control mask)] for the given games and blur levels."""
    out = []
    for _, game in bench.available(numbers):
        gt = bench.reference(game)[0]
        for blur in blurs:
            gt_codes, tracking, graded = bench.setup(game, hide, blur)
            ctrl = events.control_mask(game["events"], game["n_frames"])
            out.append((game["name"] + "/" + blur, gt, evidence.clues(tracking), graded, ctrl))
    return out


def stack(games, design, which):
    """(names, x, y) over the games. which: 'all' graded frames, 'seen' (ball tracked) or
    'live' (team model: target is home) or 'control' (live, ball tracked: target is control)."""
    xs, ys = [], []
    for _, gt, c, graded, ctrl in games:
        ok = graded & np.isin(gt, ["H", "A", "D"])
        if which in ("seen", "control"):
            ok &= c["present"]
        if which in ("live", "control"):
            ok &= gt != "D"
        names, x = design(c)
        xs.append(x[ok])
        ys.append(ctrl[ok] if which == "control" else gt[ok] == ("H" if which == "live" else "D"))
    return names, np.vstack(xs), np.concatenate(ys)


def fit(games, design, which, intercept):
    names, x, y = stack(games, design, which)
    return as_dict(names, *evidence.fit_logistic(x, y, intercept=intercept, ridge=RIDGE)), len(y)


def report(label, games, dead_model, team_model, control_model):
    for name, gt, c, graded, ctrl in games:
        dead, team = evidence.log_odds(c, dead_model, team_model)
        ok = graded & np.isin(gt, ["H", "A", "D"])
        live = ok & (gt != "D")
        unseen = ~c["present"]
        dead_acc = np.mean((dead[ok] > 0) == (gt[ok] == "D"))
        team_acc = np.mean((team[live] > 0) == (gt[live] == "H"))
        gap_team = np.mean((team[live & unseen] > 0) == (gt[live & unseen] == "H"))
        gap_dead = np.mean(dead[live & unseen] > 0)
        seen_live = live & ~unseen
        ctrl_acc = np.mean((evidence.control_log_odds(c, control_model)[seen_live] > 0)
                           == ctrl[seen_live])
        print("  {:<8} {:<10} dead acc {:.3f}  live-but-unseen called dead {:.3f}  "
              "team acc {:.3f} (ball unseen {:.3f})  ctrl acc {:.3f}".format(
                  label, name, dead_acc, gap_dead, team_acc, gap_team, ctrl_acc))


def as_dict(names, bias, weights):
    return {"names": list(names), "bias": round(float(bias), 10),
            "weights": [round(float(w), 10) for w in weights]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heldout", action="store_true",
                        help="also report the held-out game (off by default, checklist 4.10)")
    args = parser.parse_args()
    config.seed_everything()
    tuning = rows(bench.TUNING)
    motion, n_motion = fit(tuning, evidence.motion_design, "all", True)
    ball, n_ball = fit(tuning, evidence.ball_design, "seen", True)
    team_model, n_team = fit(tuning, evidence.team_design, "live", False)
    control_model, n_control = fit(tuning, evidence.control_design, "control", True)
    dead_model = {"motion": motion, "ball": ball}

    for title, model, n in (("dead, no ball (motion only)", motion, n_motion),
                            ("dead, ball tracked", ball, n_ball),
                            ("team (home +)", team_model, n_team),
                            ("control (at feet +)", control_model, n_control)):
        print("{} model, {} frames, bias {:+.3f}".format(title, n, model["bias"]))
        for name, w in zip(model["names"], model["weights"]):
            print("    {:<18}{:+.4f}".format(name, w))
    print()
    report("tune", tuning, dead_model, team_model, control_model)
    if args.heldout:
        for hide in (HIDE_BALL, 0.15, 0.35):
            report("HELDOUT@{:.0%}".format(hide), rows(bench.HELD_OUT, hide), dead_model,
                   team_model, control_model)

    with open(OUT, "w") as f:
        f.write('"""Generated by tools/fit_evidence.py; do not edit by hand.\n\n'
                "Fitted on Metrica Games 1 and 3 only (light/medium/heavy ball blur, ball hidden "
                "on {:.0%} of live frames).\n"
                "Includes the control-vs-loose model (CONTROL_MODEL), fitted on the same tuning "
                "games and degraded views.\n"
                '"""\n\n'.format(HIDE_BALL))
        f.write("DEAD_MODEL = {!r}\n\nTEAM_MODEL = {!r}\n\nCONTROL_MODEL = {!r}\n".format(
            dead_model, team_model, control_model))
    print("\nwrote", os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
