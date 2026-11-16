"""Score labelers on tuning games at every tracker level: the development view.

    python tools/scoreboard.py                                   # v1 v3 v4 on IDSSE 12 13 15
    python tools/scoreboard.py --games 1 3 12 13 15 --hides 0.15 0.25 0.35

spellF1 (and possessions predicted / reference) per game and tracker level, with the ball
hidden on the given fractions of live frames, X off, plus means per labeler. Tuning games only:
a held-out game (Metrica 2, IDSSE 11/14/16/17) is refused by bench.available before anything is
read. Nothing is written. Deterministic.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config, labelers  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

DEFAULT_GAMES = list(bench.IDSSE_TUNING)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", nargs="+", default=["v1", "v3", "v4"],
                        choices=sorted(labelers.LABELERS))
    parser.add_argument("--games", nargs="+", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--levels", nargs="+", default=list(degrade.TRACKER_LEVELS),
                        choices=list(degrade.TRACKER_LEVELS))
    parser.add_argument("--hides", nargs="+", type=float, default=[degrade.LIVE_MISSING])
    parser.add_argument("--pool", choices=sorted(bench.POOLS), default=None,
                        help="named pool instead of --games: dev (the default games) or full "
                             "(all ten external matches, incl. the spent held-out ones; "
                             "docs/refinement2_rule.md)")
    args = parser.parse_args(argv)
    config.seed_everything()
    spent = False
    if args.pool:
        args.games, spent = bench.pool(args.pool)
    games = bench.available(args.games, heldout=spent)   # refuses held-out games without --pool full
    results = {}
    for hide in args.hides:
        print("\nball hidden on {:.0%} of live frames".format(hide))
        print("{:<8}{:<10}".format("labeler", "game") + "".join(
            "{:>16}".format(level) for level in args.levels) + "{:>10}".format("mean"))
        for name in args.labeler:
            per_game = []
            for _, game in games:
                cells, scores = [], []
                for level in args.levels:
                    r = bench.score(name, game, hide, level, use_abstain=False)
                    scores.append(r["spell_f1_mean"])
                    cells.append("{:.3f} {:>4}/{:<4}".format(
                        r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
                per_game.append(scores)
                results[(hide, name, game["name"])] = scores
                print("{:<8}{:<10}".format(name, game["name"]) + "".join(
                    "{:>16}".format(c) for c in cells) + "{:>10.4f}".format(np.mean(scores)))
            per_level = np.mean(per_game, axis=0)
            print("{:<8}{:<10}".format(name, "mean") + "".join(
                "{:>16.4f}".format(s) for s in per_level) + "{:>10.4f}".format(
                    np.mean(per_game)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
