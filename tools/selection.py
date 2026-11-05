"""The one-shot comparison of the final candidates on held-out Metrica Game 2 (checklist 9.2 / 9.3).

    python tools/selection.py --dry-run     # list what would run; reads no game
    python tools/selection.py --smoke       # exercise the real scoring on the TUNING games only;
                                            # never touches Game 2, writes nothing
    python tools/selection.py               # THE one shot: score, print, write docs/selection.md
    python tools/selection.py --again       # a deliberate re-run: appends a new dated section

Game 2 is held out: nothing was ever tuned on it, and the plan is to score it ONCE, at selection,
for the final candidates only. This tool makes that one shot complete (every candidate, with and
without the X policy, at every tracker level and ball-hiding level, plus the tuning games for
reference), reproducible (same bench, fixed seeds) and written down with the commit it ran on.

How to read the result: spellF1 decides the ranking; the headline is the mean over the four
tracker levels at 25% live ball hidden, with 15% / 35% as robustness checks. macroF1 is the
official tie-break and the only number the X policy can move (the scorer fills X with the
neighbouring label before matching possessions, so spellF1 cannot change). Possession counts
show over- or under-segmentation. Differences below NOISE are not evidence.
"""

import argparse
import datetime
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import abstain, config  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

CANDIDATES = ("v1", "v3", "v4")
ABSTAIN = (False, True)
HIDES = (0.15, 0.25, 0.35)
HEADLINE_HIDE = degrade.LIVE_MISSING
LEVELS = degrade.TRACKER_LEVELS
GAMES = bench.TUNING + bench.HELD_OUT        # tuning games first, held-out last
NOISE = 0.01                                 # the bench's noise level (tools/tune_abstain.py)
REPORT = os.path.join(ROOT, "docs", "selection.md")
METRICS = ("spell_f1", "macro_f1", "n_pred", "n_gt", "n_x")


class AlreadyRun(RuntimeError):
    """The report exists: the one shot has been taken. Re-running needs --again."""


# --------------------------------------------------------------------------- what to run

def plan(candidates=CANDIDATES, abstain=ABSTAIN, hides=HIDES, levels=LEVELS, games=GAMES):
    """[(candidate, abstain_on, game, hide, level)], games outermost so each loads once."""
    return [(c, a, g, h, l)
            for g in games for h in hides for l in levels for c in candidates for a in abstain]


def score_one(game, candidate, abstain_on, hide, level):
    """Score one candidate on one degraded view of one game; the X setting is applied only here."""
    old = abstain.PARAMS.get(candidate)
    try:
        if abstain_on:
            abstain.PARAMS[candidate] = abstain.SETTING
        r = bench.score(candidate, game, hide_ball=hide, blur=level, use_abstain=abstain_on)
    finally:
        abstain.PARAMS[candidate] = old
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "n_pred": int(r["n_pred_spells"]), "n_gt": int(r["n_gt_spells"]),
            "n_x": int(r["n_abstain_on_graded"])}


def run(specs, games):
    """Score every spec in order; games = {number: loaded game}. Prints one line per scoring."""
    results = []
    print("{:<6}{:<5}{:<6}{:<7}{:<8}{:>9}{:>9}{:>7}{:>7}{:>7}".format(
        "cand", "X", "game", "hide", "level", "spellF1", "macroF1", "pred", "ref", "X-fr"))
    for candidate, abstain_on, number, hide, level in specs:
        r = score_one(games[number], candidate, abstain_on, hide, level)
        r.update(candidate=candidate, abstain=abstain_on, game=number, hide=hide, level=level)
        results.append(r)
        print("{:<6}{:<5}{:<6}{:<7.0%}{:<8}{:>9.4f}{:>9.4f}{:>7}{:>7}{:>7}".format(
            candidate, "on" if abstain_on else "off", number, hide, level,
            r["spell_f1"], r["macro_f1"], r["n_pred"], r["n_gt"], r["n_x"]))
    return results


# --------------------------------------------------------------------------- summarising

def _mean_block(rows, levels):
    """Means over rows, plus the mean spellF1 per tracker level."""
    block = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    block["by_level"] = {}
    for level in levels:
        at = [r["spell_f1"] for r in rows if r["level"] == level]
        if at:
            block["by_level"][level] = float(np.mean(at))
    return block


def summarise(results, tuning=bench.TUNING, heldout=bench.HELD_OUT[0]):
    """{(candidate, abstain): {"heldout": {hide: block}, "tuning": {hide: block}}}.

    A block holds the mean of every metric over the rows it covers (held-out: the tracker
    levels; tuning: the tuning games x the tracker levels) and the mean spellF1 per level.
    """
    levels = sorted({r["level"] for r in results}, key=lambda l: list(LEVELS).index(l)
                    if l in LEVELS else len(LEVELS))
    summary = {}
    for r in results:
        key = (r["candidate"], r["abstain"])
        summary.setdefault(key, {"heldout": {}, "tuning": {}})
    for key, out in summary.items():
        mine = [r for r in results if (r["candidate"], r["abstain"]) == key]
        for hide in sorted({r["hide"] for r in mine}):
            held = [r for r in mine if r["hide"] == hide and r["game"] == heldout]
            tune = [r for r in mine if r["hide"] == hide and r["game"] in tuning]
            if held:
                out["heldout"][hide] = _mean_block(held, levels)
            if tune:
                out["tuning"][hide] = _mean_block(tune, levels)
    return summary


def _ranked(summary, hide, abstain_on=False):
    """[(candidate, spellF1)] on the held-out game at `hide`, best first, one X mode only."""
    rows = [(c, s["heldout"][hide]["spell_f1"]) for (c, a), s in summary.items()
            if a == abstain_on and hide in s["heldout"]]
    return sorted(rows, key=lambda x: (-x[1], x[0]))


def reading(summary, hide=HEADLINE_HIDE, levels=LEVELS, noise=NOISE):
    """Plain sentences about the held-out result: leader, margin, levels, the X check, overfit."""
    lines = []
    ranked = _ranked(summary, hide)
    if not ranked:
        return ["No held-out result at {:.0%}.".format(hide)]
    leader, best = ranked[0]
    line = "Leader on held-out Game 2 at {:.0%} live ball hidden: {} (X off), spellF1 {:.3f}".format(
        hide, leader, best)
    if len(ranked) > 1:
        runner, second = ranked[1]
        margin = best - second
        line += ", ahead of {} by {:.3f}, {} the noise level ({:.2f})".format(
            runner, margin, "above" if margin >= noise else "within", noise)
        by_level = {c: s["heldout"][hide]["by_level"] for (c, a), s in summary.items()
                    if not a and hide in s["heldout"]}
        won = [l for l in levels if l in by_level[leader]
               and all(by_level[leader][l] >= by_level[c].get(l, -1) for c in by_level)]
        line += "; ahead at {} of {} tracker levels ({})".format(
            len(won), len(levels), ", ".join(won) if won else "none")
    lines.append(line + ".")

    for candidate, _ in ranked:
        off, on = summary.get((candidate, False)), summary.get((candidate, True))
        if not (off and on and hide in off["heldout"] and hide in on["heldout"]):
            continue
        a, b = off["heldout"][hide], on["heldout"][hide]
        if abs(a["spell_f1"] - b["spell_f1"]) < 1e-9:
            verdict = "X leaves spellF1 unchanged for {} (as designed)".format(candidate)
        else:
            verdict = "WARNING: X changed spellF1 for {} by {:+.4f} (it must not)".format(
                candidate, b["spell_f1"] - a["spell_f1"])
        lines.append("{}; macroF1 {:+.3f} ({:.3f} -> {:.3f}), X on {:.0f} graded frames.".format(
            verdict, b["macro_f1"] - a["macro_f1"], a["macro_f1"], b["macro_f1"], b["n_x"]))

    for candidate, held in ranked:
        tune = summary[(candidate, False)]["tuning"].get(hide)
        if tune:
            lines.append("{}: tuning-game mean {:.3f} vs held-out {:.3f} ({:+.3f}); a large drop "
                         "means the settings are overfitted to the tuning games.".format(
                             candidate, tune["spell_f1"], held, held - tune["spell_f1"]))
    return lines


# --------------------------------------------------------------------------- the report

def _cell(block, metric, fmt="{:.3f}"):
    return fmt.format(block[metric]) if block else "–"


def render(summary, hides=HIDES, levels=LEVELS, noise=NOISE, meta=None):
    """Markdown: provenance, summary table, per-level table, reading."""
    meta = meta or {}
    headline = HEADLINE_HIDE if HEADLINE_HIDE in hides else hides[len(hides) // 2]
    others = [h for h in hides if h != headline]
    out = ["## Selection run: {}, commit `{}`{}".format(
        meta.get("date", "?"), meta.get("commit", "?"),
        " **(uncommitted changes in the working tree)**" if meta.get("dirty") else ""), ""]
    out.append("Candidates scored on held-out Metrica Game 2 and, for reference, on the tuning "
               "games (1 + 3). spellF1 = mean over the tracker levels {}. X = `abstain.SETTING` "
               "= `{}`. Noise level {:.2f}: smaller differences are not evidence.".format(
                   " / ".join(levels), abstain.SETTING, noise))
    out.append("")
    out.append("### Summary (spellF1, mean over tracker levels)")
    out.append("")
    head = ["candidate", "X", "tuning @{:.0%}".format(headline),
            "**Game 2 @{:.0%}**".format(headline)]
    head += ["Game 2 @{:.0%}".format(h) for h in others]
    head += ["macroF1 Game 2 @{:.0%}".format(headline),
             "possessions Game 2 @{:.0%} (pred / ref)".format(headline), "X frames"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    keys = sorted(summary, key=lambda k: (-(summary[k]["heldout"].get(headline, {})
                                             .get("spell_f1", -1)), k[0], k[1]))
    for candidate, abstain_on in keys:
        s = summary[(candidate, abstain_on)]
        held = s["heldout"].get(headline)
        row = [candidate, "on" if abstain_on else "off",
               _cell(s["tuning"].get(headline), "spell_f1"),
               "**{}**".format(_cell(held, "spell_f1"))]
        row += [_cell(s["heldout"].get(h), "spell_f1") for h in others]
        row += [_cell(held, "macro_f1"),
                "{} / {}".format(_cell(held, "n_pred", "{:.0f}"), _cell(held, "n_gt", "{:.0f}")),
                _cell(held, "n_x", "{:.0f}")]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("### Game 2 by tracker level (spellF1 @{:.0%})".format(headline))
    out.append("")
    out.append("| candidate | X | " + " | ".join(levels) + " |")
    out.append("|---|---|" + "---|" * len(levels))
    for candidate, abstain_on in keys:
        held = summary[(candidate, abstain_on)]["heldout"].get(headline)
        cells = ["{:.3f}".format(held["by_level"][l]) if held and l in held["by_level"] else "–"
                 for l in levels]
        out.append("| {} | {} | {} |".format(candidate, "on" if abstain_on else "off",
                                             " | ".join(cells)))
    out.append("")
    out.append("### Reading")
    out.append("")
    out += ["- " + line for line in reading(summary, headline, levels, noise)]
    out.append("")
    out.append("**Decision:** _(fill in: which variant, X on or off, and why)_")
    out.append("")
    return "\n".join(out)


def write_report(path, text, again=False):
    """Create the report, or append a section to it if `again`; never overwrite silently."""
    exists = os.path.exists(path)
    if exists and not again:
        raise AlreadyRun("{} exists: the one-shot selection has been run. Use --again to append "
                         "a new section (and say in the decision why a second run was needed)."
                         .format(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a" if exists else "w", encoding="utf-8") as handle:
        if not exists:
            handle.write("# Selection on held-out Metrica Game 2\n\nWritten by "
                         "`tools/selection.py`. Each run appends a section; never edit the "
                         "numbers by hand.\n\n")
        else:
            handle.write("\n")
        handle.write(text)


def provenance():
    def git(*args):
        try:
            return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"date": datetime.date.today().isoformat(),
            "commit": git("rev-parse", "--short", "HEAD") or "unknown",
            "dirty": bool(git("status", "--porcelain"))}


# --------------------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES),
                        help="labelers to compare (default: {})".format(" ".join(CANDIDATES)))
    parser.add_argument("--no-x", action="store_true", help="skip the 'X policy on' variants")
    parser.add_argument("--hides", nargs="+", type=float, default=list(HIDES),
                        help="live-ball hiding levels (default: {})".format(
                            " ".join(str(h) for h in HIDES)))
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(degrade.BLUR),
                        help="tracker levels (default: {})".format(" ".join(LEVELS)))
    parser.add_argument("--no-tuning", action="store_true",
                        help="score only the held-out game (no tuning-game reference columns)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run and stop; reads no game")
    parser.add_argument("--smoke", action="store_true",
                        help="run the real scoring on the tuning games only and write no report "
                             "(a rehearsal that never touches the held-out game)")
    parser.add_argument("--again", action="store_true",
                        help="append a new section to an existing report (a second shot)")
    parser.add_argument("--out", default=REPORT, help="report path (default: docs/selection.md)")
    args = parser.parse_args(argv)

    abstain_modes = (False,) if args.no_x else ABSTAIN
    if args.smoke:
        games = bench.TUNING
    else:
        games = bench.HELD_OUT if args.no_tuning else bench.TUNING + bench.HELD_OUT
    specs = plan(args.candidates, abstain_modes, tuple(args.hides), tuple(args.levels), games)
    exists = os.path.exists(args.out)

    print("{} scorings: candidates {}; X {}; hide levels {}; tracker levels {}; games {} "
          "(held out: {}).".format(len(specs), " ".join(args.candidates),
                                   "off + on" if len(abstain_modes) == 2 else "off",
                                   " ".join("{:.0%}".format(h) for h in args.hides),
                                   " ".join(args.levels),
                                   " ".join(str(g) for g in games), bench.HELD_OUT[0]))
    print("report: {}{}".format(
        "none (smoke run)" if args.smoke else args.out,
        " (exists; --again needed)" if exists and not args.smoke else ""))
    if args.dry_run:
        return 0
    if exists and not args.again and not args.smoke:
        print("STOP: {} already exists, so the one shot has been taken. Re-run with --again only "
              "if the team agrees a second run is needed.".format(args.out))
        return 1

    config.seed_everything()
    loaded = dict(bench.available(games))
    missing = [g for g in games if g not in loaded]
    if missing:
        print("STOP: game(s) {} not available; run tools/fetch_metrica.py first.".format(missing))
        return 1
    results = run(specs, loaded)
    summary = summarise(results, tuning=bench.TUNING, heldout=bench.HELD_OUT[0])
    text = render(summary, tuple(args.hides), tuple(args.levels), NOISE, provenance())
    print("\n" + text)
    if args.smoke:
        print("smoke run: nothing written, held-out game untouched.")
        return 0
    write_report(args.out, text, again=args.again)
    print("written: {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
