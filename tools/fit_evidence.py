"""Fit the two evidence models (checklist 5.10) on the external TUNING games only.

    python tools/fit_evidence.py              # fits, prints a report, writes espadas/evidence_models.py

Training rows: every graded frame of Metrica Games 1 and 3, once per tracker level (light,
medium, heavy, smooth; degrade.TRACKER_LEVELS: ball blur plus player position error), with the
ball hidden on degrade.LIVE_MISSING of live frames (the same degraded views the labelers are
tuned on).
- dead model, in two parts (see evidence.DEAD_SPLIT), both reference D vs H/A with an intercept:
  "motion" (players only) fitted on all graded frames, "ball" (players + ball) on the graded
  frames where the ball is tracked.
- team model: reference H vs A on live frames only, no intercept (every team clue is
  home-minus-away, so the model is symmetric between the teams).
Labels come from the games' events by code (events.py). The held-out game (Metrica 2) is never
fitted on, and is not even reported unless --heldout is given (tools/selection.py scores it once,
at selection); --skip-heldout is still accepted and changes nothing. Our match is never touched.
Deterministic.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config, evidence  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

OUT = os.path.join(ROOT, "espadas", "evidence_models.py")
HIDE_BALL = degrade.LIVE_MISSING
RIDGE = 1.0


BLURS = degrade.TRACKER_LEVELS   # fit on every tracker level (checklist 6.6, 6.7), as tune.py


def rows(numbers, hide=HIDE_BALL, blurs=BLURS, heldout=False):
    """[(name, reference labels, clues, graded mask)] for the given games and blur levels."""
    out = []
    for _, game in bench.available(numbers, heldout=heldout):
        gt = bench.reference(game)[0]
        for blur in blurs:
            gt_codes, tracking, graded = bench.setup(game, hide, blur)
            out.append((game["name"] + "/" + blur, gt, evidence.clues(tracking), graded))
    return out


def stack(games, design, which):
    """(names, x, y) over the games. which: 'all' graded frames, 'seen' (ball tracked) or
    'live' (team model: target is home)."""
    xs, ys = [], []
    for _, gt, c, graded in games:
        ok = graded & np.isin(gt, ["H", "A", "D"])
        if which == "seen":
            ok &= c["present"]
        if which == "live":
            ok &= gt != "D"
        names, x = design(c)
        xs.append(x[ok])
        ys.append(gt[ok] == ("H" if which == "live" else "D"))
    return names, np.vstack(xs), np.concatenate(ys)


def fit(games, design, which, intercept):
    names, x, y = stack(games, design, which)
    return as_dict(names, *evidence.fit_logistic(x, y, intercept=intercept, ridge=RIDGE)), len(y)


def report(label, games, dead_model, team_model):
    for name, gt, c, graded in games:
        dead, team = evidence.log_odds(c, dead_model, team_model)
        ok = graded & np.isin(gt, ["H", "A", "D"])
        live = ok & (gt != "D")
        unseen = ~c["present"]
        dead_acc = np.mean((dead[ok] > 0) == (gt[ok] == "D"))
        team_acc = np.mean((team[live] > 0) == (gt[live] == "H"))
        gap_team = np.mean((team[live & unseen] > 0) == (gt[live & unseen] == "H"))
        gap_dead = np.mean(dead[live & unseen] > 0)
        print("  {:<8} {:<10} dead acc {:.3f}  live-but-unseen called dead {:.3f}  "
              "team acc {:.3f} (ball unseen {:.3f})".format(label, name, dead_acc, gap_dead,
                                                          team_acc, gap_team))


def as_dict(names, bias, weights):
    return {"names": list(names), "bias": round(float(bias), 10),
            "weights": [round(float(w), 10) for w in weights]}


def fit_models(games):
    """(dead_model, team_model, (n_motion, n_ball, n_team)) fitted on rows() of some games.
    Shared with tools/logo.py, which fits on one game at a time."""
    motion, n_motion = fit(games, evidence.motion_design, "all", True)
    ball, n_ball = fit(games, evidence.ball_design, "seen", True)
    team_model, n_team = fit(games, evidence.team_design, "live", False)
    return {"motion": motion, "ball": ball}, team_model, (n_motion, n_ball, n_team)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heldout", action="store_true",
                        help="also report on the held-out game (off by default: "
                             "tools/selection.py scores it once, at selection)")
    parser.add_argument("--skip-heldout", action="store_true",
                        help="do not report on the held-out game (the default now; kept so old "
                             "commands still run; wins over --heldout)")
    args = parser.parse_args(argv)
    config.seed_everything()
    tuning = rows(bench.TUNING)
    dead_model, team_model, (n_motion, n_ball, n_team) = fit_models(tuning)
    motion, ball = dead_model["motion"], dead_model["ball"]

    for title, model, n in (("dead, no ball (motion only)", motion, n_motion),
                            ("dead, ball tracked", ball, n_ball),
                            ("team (home +)", team_model, n_team)):
        print("{} model, {} frames, bias {:+.3f}".format(title, n, model["bias"]))
        for name, w in zip(model["names"], model["weights"]):
            print("    {:<18}{:+.4f}".format(name, w))
    print()
    report("tune", tuning, dead_model, team_model)
    for hide in (HIDE_BALL, 0.15, 0.35) if args.heldout and not args.skip_heldout else ():
        report("HELDOUT@{:.0%}".format(hide), rows(bench.HELD_OUT, hide, heldout=True),
               dead_model, team_model)

    with open(OUT, "w") as f:
        f.write('"""Generated by tools/fit_evidence.py; do not edit by hand.\n\n'
                "Fitted on Metrica Games 1 and 3 only (light/medium/heavy/smooth tracker levels: ball blur + "
                "player error, ball hidden "
                "on {:.0%} of live frames).\n"
                '"""\n\n'.format(HIDE_BALL))
        f.write("DEAD_MODEL = {!r}\n\nTEAM_MODEL = {!r}\n".format(dead_model, team_model))
    print("\nwrote", os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
