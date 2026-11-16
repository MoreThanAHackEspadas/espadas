"""Diagnostic (refinement 2, step 3): would a non-linear evidence model beat the logistic one?

    python tools/evidence_diag.py                 # leave-one-game-out over bench.DEV_TUNING

For every game left out, the team model (home vs away on live frames) and the two dead models
are fitted on the other games, twice: the shipped way (evidence.fit_logistic) and with a small
gradient-boosted model (scikit-learn HistGradientBoostingClassifier, fixed seed) on the same
clue columns. Reported on the game left out: per-frame log-loss and accuracy, and for the team
model the accuracy on frames where the ball is not tracked. Nothing is written; scikit-learn is
needed by this tool only, never by the build. External tuning games only. Deterministic.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from espadas import config, evidence  # noqa: E402
from espadas.external import bench  # noqa: E402
import fit_evidence  # noqa: E402

STEP = 5          # boosted model: fit on every STEP-th frame (neighbouring frames are near copies)
BOOST = dict(max_depth=3, max_iter=200, learning_rate=0.1, l2_regularization=1.0,
             min_samples_leaf=200, random_state=0)
MODELS = (("dead, no ball", "motion_design", "all", True),
          ("dead, ball", "ball_design", "seen", True),
          ("team", "team_design", "live", False))


def logloss(z, y):
    z = np.clip(z, -30, 30)
    return float(np.mean(np.logaddexp(0.0, -np.where(y, z, -z))))


def boosted(x, y, symmetric):
    from sklearn.ensemble import HistGradientBoostingClassifier
    x, y = x[::STEP], y[::STEP]
    if symmetric:        # team clues are home-minus-away: add the mirrored match, no home bias
        x, y = np.vstack([x, -x]), np.concatenate([y, ~y])
    model = HistGradientBoostingClassifier(**BOOST).fit(x, y)
    if symmetric:
        return lambda q: 0.5 * (model.decision_function(q) - model.decision_function(-q))
    return model.decision_function


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING))
    args = parser.parse_args(argv)
    config.seed_everything()
    by_game = {}
    for number in args.games:
        rows = fit_evidence.rows([number])
        by_game[number] = (rows[0][0].split("/")[0], rows)
    print("left-out game: log-loss / accuracy, logistic -> boosted "
          "(team: also accuracy with the ball not tracked)")
    totals = {}
    for number in args.games:
        name, test = by_game[number]
        train = [r for n in args.games if n != number for r in by_game[n][1]]
        for title, design, which, intercept in MODELS:
            fn = getattr(evidence, design)
            _, x, y = fit_evidence.stack(train, fn, which)
            _, xt, yt = fit_evidence.stack(test, fn, which)
            ok, okt = np.isfinite(x).all(axis=1), np.isfinite(xt).all(axis=1)
            bias, w = evidence.fit_logistic(x[ok], y[ok], intercept=intercept,
                                            ridge=fit_evidence.RIDGE)
            z_lin = bias + xt[okt] @ np.asarray(w)
            z_gb = boosted(x[ok], y[ok], symmetric=not intercept)(xt[okt])
            yt = yt[okt]
            cells = [logloss(z_lin, yt), logloss(z_gb, yt),
                     np.mean((z_lin > 0) == yt), np.mean((z_gb > 0) == yt)]
            line = "  {:<10}{:<14} loss {:.4f} -> {:.4f}   acc {:.4f} -> {:.4f}".format(
                name, title, *cells)
            if which == "live":
                unseen = np.concatenate([
                    ~c["present"][g & np.isin(gt, ["H", "A"])] for _, gt, c, g in test])[okt]
                cells += [np.mean((z_lin > 0)[unseen] == yt[unseen]),
                          np.mean((z_gb > 0)[unseen] == yt[unseen])]
                line += "   unseen {:.4f} -> {:.4f}".format(*cells[4:])
            totals.setdefault(title, []).append(cells)
            print(line, flush=True)
    for title, cells in totals.items():
        m = np.mean(cells, axis=0)
        line = "  {:<10}{:<14} loss {:.4f} -> {:.4f}   acc {:.4f} -> {:.4f}".format(
            "MEAN", title, *m[:4])
        if len(m) > 4:
            line += "   unseen {:.4f} -> {:.4f}".format(*m[4:])
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
