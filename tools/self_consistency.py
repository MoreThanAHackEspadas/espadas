"""Diagnostic only (checklist 4.7): what a labeler claims about OUR match, without any truth.

    python tools/self_consistency.py --labeler v0
    python tools/self_consistency.py --tracking     # input quality, ours vs the external games

Prints possession count, mean duration, home share and label mix on graded frames, next to the
same numbers for the external reference labels, so implausible output (900 possessions of 5 s)
is visible early. --tracking instead compares the *input* (players tracked, ball seen, how close
the nearest player is, ball jitter, touches found), to see where our tracking differs from the
external games (checklist 6.6). Never use these numbers to choose settings: tuning happens on
external games only (organisers' clarification). Prints aggregates only, no per-frame data.
The external rows are the tuning games; --heldout adds the held-out game (off by default:
tools/selection.py scores it once, at selection).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import atd, config, labelers, touches, v3, write  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

pm = config.pm


def stats(labels):
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    s = pm.possession_stats(codes, runs)
    graded = codes != pm.X
    mix = {k: float(np.mean(codes[graded] == getattr(pm, k))) for k in "HAD"}
    return s["n_spells"], s["mean_duration_s"], s["home_share"], mix, int(graded.sum())


def line(name, labels):
    n, mean, share, mix, graded = stats(labels)
    per90 = n / (graded / config.FPS / 60) * 90
    print("{:<22}{:>7}{:>9.0f}{:>9.1f}{:>8.2f}{:>7.0%}{:>7.0%}{:>7.0%}".format(
        name, n, per90, mean, share, mix["H"], mix["A"], mix["D"]))


def tracking_line(name, tracking, graded):
    """One row of input-quality aggregates over graded frames."""
    f = v3.features(tracking)
    p = v3.PARAMS
    players = tracking["players"][graded]
    tracked = np.sum(~np.isnan(players[..., 0]), axis=1)
    present = f["present"][graded]
    near = f["near"][graded & f["present"]]
    kink = f["kinks"][int(p["kink_half"])]
    jitter = f["kinks"][touches.KINK_HALF][graded & f["present"]]
    t = touches.detect(kink, f["near"], f["slot"], f["team_of_slot"],
                       p["kink_threshold"], p["touch_reach"])
    ok = touches.confirmed(t, f["dist"], int(p["chain_gap"]), p["hold_reach"],
                           int(p["hold_frames"]))
    in_graded = graded[t["frame"]]
    minutes = graded.sum() / tracking.get("fps", config.FPS) / 60
    print("{:<16}{:>8.1f}{:>8.0%}{:>9.2f}{:>8.0%}{:>8.0%}{:>9.2f}{:>9.0f}{:>8.0%}".format(
        name, float(np.mean(tracked)), float(np.mean(present)), float(np.median(near)),
        float(np.mean(near <= 1.0)), float(np.mean(near <= 2.0)), float(np.median(jitter)),
        in_graded.sum() / minutes * 90, float(np.mean(ok[in_graded])) if in_graded.any() else 0))


def external_games(heldout=False):
    """Game numbers to show: the tuning games, plus the held-out one only if asked."""
    return sorted(bench.TUNING + (bench.HELD_OUT if heldout else ()))


def tracking_table(data_dir, heldout=False):
    print("{:<16}{:>8}{:>8}{:>9}{:>8}{:>8}{:>9}{:>9}{:>8}".format(
        "", "players", "ball", "near m", "<=1m", "<=2m", "jitter", "tch/90'", "conf"))
    print("{:<16}{:>8}{:>8}{:>9}{:>8}{:>8}{:>9}{:>9}{:>8}".format(
        "", "tracked", "seen", "median", "", "", "m/s", "", ""))
    for n, game in bench.available(external_games(heldout), heldout=heldout):
        _, tracking, graded = bench.setup(game, degrade.LIVE_MISSING)
        tracking_line(game["name"], tracking, graded)
    tracking = atd.load_tracking(data_dir)
    graded = config.half_of_frame(len(tracking["ball"]), config.HALF_BLOCKS) > 0
    tracking_line("BAR-ATH", tracking, graded)
    print("external games with {:.0%} of live ball hidden, as the labelers see them; jitter = "
          "median ball velocity change (touches.kink) on frames with a ball".format(
              degrade.LIVE_MISSING))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", default="v0", choices=sorted(labelers.LABELERS))
    parser.add_argument("--data-dir")
    parser.add_argument("--tracking", action="store_true",
                        help="compare input quality instead of a labeler's output")
    parser.add_argument("--heldout", action="store_true",
                        help="also show the held-out game (off by default: tools/selection.py "
                             "scores it once, at selection)")
    args = parser.parse_args(argv)
    config.seed_everything()
    if args.tracking:
        tracking_table(args.data_dir, args.heldout)
        return 0

    print("{:<22}{:>7}{:>9}{:>9}{:>8}{:>7}{:>7}{:>7}".format(
        "", "spells", "per 90'", "mean s", "home", "H", "A", "D"))
    for n, game in bench.available(external_games(args.heldout), heldout=args.heldout):
        line("ref " + game["name"], bench.reference(game)[0])
    tracking = atd.load_tracking(args.data_dir)
    line("{} on BAR-ATH".format(args.labeler),
         write.finalise(labelers.LABELERS[args.labeler](tracking)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
