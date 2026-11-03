"""Score any labeler on external games with the official metric. The only judge for settings.

    python -m espadas.external.bench                  # v0 on every available game
    python -m espadas.external.bench --labeler v0 --games 1 2 3

Reference labels come from each game's events by code (events.py). The labeler sees only the
tracking (as it would on our match) plus the match-level frame layout (halves, kickoffs), the
external equivalent of games.py. Frames outside the graded halves are set to X, as finalise()
does for our match. Scored with the vendored official possession_metrics.evaluate().
"""

import argparse
import os
import sys

import numpy as np

from .. import config, labelers
from . import degrade
from . import events as ev
from . import metrica_csv, metrica_epts

pm = config.pm

GAMES = {1: metrica_csv.load_game, 2: metrica_csv.load_game, 3: metrica_epts.load_game}
# Agreed split (checklist 4.4): tune on 1 and 3, hold out 2 for the final comparison only.
TUNING, HELD_OUT = (1, 3), (2,)


def reference(game):
    labels, half_blocks, kickoffs = ev.labels_from_events(game["events"], game["n_frames"])
    return labels, half_blocks, kickoffs


def score(labeler_name, game, hide_ball=None):
    gt, half_blocks, kickoffs = reference(game)
    tracking = dict(game["tracking"], half_blocks=half_blocks, kickoffs=kickoffs,
                    fps=game["fps"], pitch=game["pitch"])
    if hide_ball is not None:
        graded = config.half_of_frame(game["n_frames"], half_blocks) > 0
        tracking["ball"], _ = degrade.hide_ball(tracking["ball"], tracking["players"], graded,
                                                hide_ball)
    pred = np.asarray(labelers.LABELERS[labeler_name](tracking), dtype="<U1").copy()
    pred[config.half_of_frame(game["n_frames"], half_blocks) == 0] = "X"
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))


def available(numbers):
    out = []
    for n in numbers:
        if n not in GAMES:
            print("game {}: no reader yet, skipped".format(n))
            continue
        try:
            out.append((n, GAMES[n](n)))
        except (IOError, OSError) as error:
            print("game {}: not downloaded ({}), skipped".format(n, error))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", nargs="+", default=["v0"], choices=sorted(labelers.LABELERS))
    parser.add_argument("--games", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--hide-ball", type=float, default=0.41, metavar="FRAC",
                        help="hide ball bursts until this fraction of graded frames has no ball "
                             "(default 0.41, like our match; pass a negative value to disable)")
    args = parser.parse_args(argv)
    config.seed_everything()

    games = available(args.games)
    print("ball hidden to {} missing on graded frames".format(
        "{:.0%}".format(args.hide_ball) if args.hide_ball >= 0 else "no extra"))
    print("\n{:<10}{:<10}{:<9}{:>9}{:>10}{:>9}{:>8}{:>8}".format(
        "labeler", "game", "split", "spellF1", "spell@.5", "macroF1", "n_pred", "n_gt"))
    for name in args.labeler:
        for n, game in games:
            r = score(name, game, args.hide_ball if args.hide_ball >= 0 else None)
            split = "tune" if n in TUNING else "HELDOUT"
            print("{:<10}{:<10}{:<9}{:>9.4f}{:>10.4f}{:>9.4f}{:>8}{:>8}".format(
                name, game["name"], split, r["spell_f1_mean"], r["spell_f1_at_headline"],
                r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
