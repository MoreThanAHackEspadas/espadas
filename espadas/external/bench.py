"""Score any labeler on external games with the official metric. The only judge for settings.

    python -m espadas.external.bench                  # v0 on the tuning games (1 and 3)
    python -m espadas.external.bench --labeler v1 v3 v4 --games 1 3

Reference labels come from each game's events by code (events.py). The labeler sees only the
tracking (as it would on our match) plus the match-level frame layout (halves, kickoffs), the
external equivalent of games.py. Frames outside the graded halves are set to X, as finalise()
does for our match. Scored with the vendored official possession_metrics.evaluate().

The held-out game (HELD_OUT) is scored once, for the final candidates, by tools/selection.py.
The game readers (metrica_csv, metrica_epts) refuse a held-out game unless the caller passes
heldout=True. available() checks the whole list first, so it refuses before reading anything,
and passes the opt-in on to the readers only when it was given one; on this command line that
is --heldout.
"""

import argparse
import os
import sys

import numpy as np

from .. import abstain, config, labelers, v2
from . import degrade
from . import events as ev
from . import idsse, metrica_csv, metrica_epts
# Agreed split (checklist 4.4): tune on 1 and 3, hold out 2 for the final comparison only.
# Defined in the package so the readers enforce the same rule.
from . import (ALL_HELD_OUT, HELD_OUT, IDSSE_HELD_OUT, IDSSE_TUNING, TUNING,  # noqa: F401
               HeldOutError, check_split)

pm = config.pm

GAMES = {1: metrica_csv.load_game, 2: metrica_csv.load_game, 3: metrica_epts.load_game}
GAMES.update({n: idsse.load_game for n in idsse.MATCHES})   # 11-17: DFL / Sportec IDSSE


def reference(game):
    """(labels, half_blocks, kickoffs). Metrica: from the events by code (events.py); IDSSE: the
    reader builds it from the provider's per-frame ball status and possession flags."""
    if game.get("reference") is not None:
        return game["reference"]
    labels, half_blocks, kickoffs = ev.labels_from_events(game["events"], game["n_frames"])
    return labels, half_blocks, kickoffs


def setup(game, hide_ball=None, blur=degrade.BLUR_DEFAULT):
    """(gt codes, labeler input, graded mask) for one game: reference labels, the tracking the
    labeler sees (ball blurred like an automatic tracker, then hidden in live play, if asked;
    players given tracker-like position error) and which frames are scored. blur is a tracker
    level, a key of degrade.BLUR / degrade.PLAYER_ERROR."""
    gt, half_blocks, kickoffs = reference(game)
    tracking = dict(game["tracking"], half_blocks=half_blocks, kickoffs=kickoffs,
                    fps=game["fps"], pitch=game["pitch"])
    graded = config.half_of_frame(game["n_frames"], half_blocks) > 0
    if degrade.BLUR[blur] is not None:
        tracking["ball"] = degrade.blur_ball(tracking["ball"], *degrade.BLUR[blur])
    if hide_ball is not None:
        live = (gt == "H") | (gt == "A")
        tracking["ball"], _ = degrade.hide_ball(tracking["ball"], tracking["players"], live,
                                                hide_ball)
    if degrade.PLAYER_ERROR.get(blur, 0.0) > 0:          # 6.7: automatic-tracker player error
        tracking["players"] = degrade.jitter_players(tracking["players"],
                                                     degrade.PLAYER_ERROR[blur])
    return pm.label_codes(gt), tracking, graded


def evaluate(gt_codes, labels, graded):
    pred = np.asarray(labels, dtype="<U1").copy()
    pred[~graded] = "X"
    return pm.evaluate(gt_codes, pm.label_codes(pred))


def score(labeler_name, game, hide_ball=None, blur=degrade.BLUR_DEFAULT, use_abstain=None):
    """Official metrics of one labeler on one degraded view of one game.

    use_abstain follows espadas.run (abstain.resolve): None = abstain.PARAMS[labeler] decides, as
    in the no-flag build; True / False force the X policy on / off."""
    gt_codes, tracking, graded = setup(game, hide_ball, blur)
    labels = np.asarray(labelers.LABELERS[labeler_name](tracking), dtype="<U1").copy()
    setting = abstain.resolve(labeler_name, use_abstain)
    if setting is not None:              # Phase 8.4, applied after the ungraded X as in run.py
        labels[~graded] = "X"
        f = v2.features(tracking)
        labels = abstain.apply(labels, f["present"], f["team_lo"], setting)
    return evaluate(gt_codes, labels, graded)


def available(numbers, heldout=False):
    """[(number, game)] for the games that can be read; the others are reported and skipped.

    A held-out game raises HeldOutError before anything is read, unless heldout=True; the readers
    get the same opt-in, so they refuse it too without one."""
    numbers = list(numbers)
    check_split(numbers, heldout)
    out = []
    for n in numbers:
        if n not in GAMES:
            print("game {}: no reader yet, skipped".format(n))
            continue
        try:
            out.append((n, GAMES[n](n, heldout=heldout)))
        except (IOError, OSError) as error:
            print("game {}: not downloaded ({}), skipped".format(n, error))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", nargs="+", default=["v0"], choices=sorted(labelers.LABELERS))
    parser.add_argument("--games", nargs="+", type=int, default=list(TUNING),
                        help="games to score (default: the tuning games {}; held-out {} needs "
                             "--heldout)".format(" ".join(map(str, TUNING)),
                                                 " ".join(map(str, HELD_OUT))))
    parser.add_argument("--heldout", action="store_true",
                        help="allow the held-out game(s) in --games. Held-out scoring belongs "
                             "to tools/selection.py (once, at selection)")
    parser.add_argument("--hide-ball", type=float, default=degrade.LIVE_MISSING, metavar="FRAC",
                        help="hide ball bursts until this fraction of LIVE frames has no ball "
                             "(default {}; negative disables)".format(degrade.LIVE_MISSING))
    parser.add_argument("--blur", default=degrade.BLUR_DEFAULT, choices=list(degrade.BLUR),
                        help="ball blur level, like an automatic tracker (default {})".format(
                            degrade.BLUR_DEFAULT))
    parser.add_argument("--abstain", action="store_true",
                        help="force the X policy on graded frames on, as `espadas.run "
                             "--abstain` (abstain.PARAMS[labeler], or abstain.SETTING if None). "
                             "Without a flag, abstain.PARAMS decides, as in the plain build")
    parser.add_argument("--no-abstain", action="store_true",
                        help="force the X policy off (wins over --abstain)")
    args = parser.parse_args(argv)
    held = sorted(set(args.games) & set(ALL_HELD_OUT))
    if held and not args.heldout:
        parser.error("game {} is held out: it is scored once, at selection, by "
                     "tools/selection.py. Score the tuning games ({}) instead, or add --heldout "
                     "only if the team agreed to score the held-out game here.".format(
                         " ".join(map(str, held)), " ".join(map(str, TUNING))))
    config.seed_everything()

    games = available(args.games, heldout=args.heldout)
    force = False if args.no_abstain else True if args.abstain else None
    print("ball blur {}, hidden to {} missing on live frames".format(
        args.blur, "{:.0%}".format(args.hide_ball) if args.hide_ball >= 0 else "no extra"))
    print("\n{:<10}{:<10}{:<9}{:>9}{:>10}{:>9}{:>8}{:>8}{:>8}".format(
        "labeler", "game", "split", "spellF1", "spell@.5", "macroF1", "n_pred", "n_gt", "n_X"))
    for name in args.labeler:
        for n, game in games:
            r = score(name, game, args.hide_ball if args.hide_ball >= 0 else None, args.blur,
                      force)
            split = "tune" if n in TUNING + IDSSE_TUNING else "HELDOUT"
            print("{:<10}{:<10}{:<9}{:>9.4f}{:>10.4f}{:>9.4f}{:>8}{:>8}{:>8}".format(
                name, game["name"], split, r["spell_f1_mean"], r["spell_f1_at_headline"],
                r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], r["n_abstain_on_graded"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
