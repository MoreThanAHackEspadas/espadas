"""The one-shot IDSSE held-out comparison (docs/idsse_comparison_rule.md): baseline vs challenger.

    python tools/idsse_compare.py --dry-run              # what would run; reads no game
    python tools/idsse_compare.py --smoke [--challenger COMMIT LABELER]
                                                         # the real machinery on the IDSSE
                                                         # TUNING matches; writes nothing
    python tools/idsse_compare.py                        # THE one shot -> docs/idsse_comparison.md

Each candidate runs from its own git worktree at its own commit (made in a temporary folder
outside the repository and removed afterwards), so the baseline is exactly main@9ebf35d's code,
evidence models and settings, and the challenger exactly what was named. Both are given the same
degraded tracking, made here by this checkout's bench (bench.setup: ball blur + player error per
tracker level, ball hidden on a share of live frames), and both are scored here, against this
checkout's IDSSE reference. The tool refuses to run if the reference or degradation code differs
from the pre-registered commit (REFERENCE_COMMIT), so the target cannot move.

Candidates are labelers called as `espadas.labelers.LABELERS[name](tracking)` with no X policy
(rule: X off), the same call bench.score makes with use_abstain=False.

The challenger is named in docs/idsse_comparison.md, on a line `Challenger: <commit> <labeler>`,
committed before the run (rule 6). The once-rule, like tools/selection.py: just before the first
held-out scoring a dated "run started" marker is appended to docs/idsse_comparison.md; a second
run refuses once that marker exists, unless --again. A real run also refuses with uncommitted
changes to tracked files. --smoke and --dry-run never write anything.
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

BASELINE = ("9ebf35d", "v4")                 # rule 1: main as frozen on 10 Nov
REFERENCE_COMMIT = "48a5a8c"                 # rule 3: the IDSSE reference as pre-registered
FROZEN_FILES = ("espadas/external/idsse.py", "espadas/external/degrade.py")
REPORT = os.path.join(ROOT, "docs", "idsse_comparison.md")
MARKER = "## Run started"
HEADLINE_HIDE = 0.25
HIDES = (0.25, 0.15, 0.35)
LEVELS = degrade.TRACKER_LEVELS
BAND = config.pm.SPELL_NOISE_BAND            # 0.04, rule 4a
TIE = 1e-9

WORKER = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from espadas import config, labelers
config.seed_everything()
fn = labelers.LABELERS[sys.argv[2]]
for line in sys.stdin:
    src, dst = line.rstrip("\n").split("\t")
    with np.load(src) as z:
        t = {k: z[k] for k in z.files}
    t["half_blocks"] = [tuple(int(v) for v in r) for r in t["half_blocks"]]
    t["kickoffs"] = tuple(int(v) for v in t["kickoffs"])
    t["fps"] = int(t["fps"]); t["pitch"] = tuple(float(v) for v in t["pitch"])
    np.save(dst, np.asarray(fn(t), dtype="<U1"))
    print("ok", flush=True)
'''


def git(*args, cwd=ROOT):
    return subprocess.run(["git"] + list(args), cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def dirty():
    return [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]


def has_commit(commit):
    """False in a clone without that commit (e.g. CI's shallow checkout)."""
    return subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT,
                          capture_output=True).returncode == 0


def frozen_ok():
    """[] if the reference / degradation code is as pre-registered, else the changed files.
    Raises RuntimeError if REFERENCE_COMMIT is not in this clone (cannot check: not "changed")."""
    if not has_commit(REFERENCE_COMMIT):
        raise RuntimeError("commit {} is not in this clone (shallow?): run `git fetch "
                           "--unshallow` first".format(REFERENCE_COMMIT))
    return [f for f in FROZEN_FILES
            if subprocess.run(["git", "diff", "--quiet", REFERENCE_COMMIT, "--", f],
                              cwd=ROOT).returncode != 0]


def named_challenger(text):
    """(commit, labeler) from the report's `Challenger: <commit> <labeler>` line, or None."""
    m = re.search(r"^Challenger:\s*`?([0-9a-f]{7,40})`?\s+`?(v\d+)`?\s*$", text, re.M)
    return (m.group(1), m.group(2)) if m else None


class Candidate:
    """A labeler at a commit, run in its own worktree by a persistent worker process."""

    def __init__(self, tmp, tag, commit, labeler):
        self.tag, self.commit, self.labeler = tag, commit, labeler
        self.path = os.path.join(tmp, tag)
        git("worktree", "add", "--detach", self.path, commit)
        self.full = git("rev-parse", "HEAD", cwd=self.path)
        script = os.path.join(tmp, "worker.py")
        with open(script, "w") as f:
            f.write(WORKER)
        env = dict(os.environ, PYTHONHASHSEED="0")
        self.proc = subprocess.Popen([sys.executable, "-I", script, self.path, labeler],
                                     cwd=self.path, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env=env)

    def label(self, src, dst):
        self.proc.stdin.write("{}\t{}\n".format(src, dst))
        self.proc.stdin.flush()
        if self.proc.stdout.readline().strip() != "ok":
            raise RuntimeError("{} worker failed".format(self.tag))
        return np.load(dst)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        git("worktree", "remove", "--force", self.path)


def save_tracking(path, tracking):
    np.savez(path, players=tracking["players"], ball=tracking["ball"],
             team_of_slot=tracking["team_of_slot"],
             half_blocks=np.array(tracking["half_blocks"]),
             kickoffs=np.array(tracking["kickoffs"]), fps=np.array(tracking["fps"]),
             pitch=np.array(tracking["pitch"]))


def decide(scores, games, levels=LEVELS):
    """Rule 4 (a)-(d) on {(cand, game, level, hide): spellF1}. cand in ('base', 'chal').
    Returns (adopt, reasons) where reasons lists each check with its numbers."""
    def mean(cand, hide, game=None, level=None):
        return float(np.mean([scores[(cand, g, l, hide)] for g in games for l in levels
                              if (game is None or g == game) and (level is None or l == level)]))
    h = HEADLINE_HIDE
    margin = mean("chal", h) - mean("base", h)
    by_game = sum(mean("chal", h, game=g) > mean("base", h, game=g) + TIE for g in games)
    by_level = sum(mean("chal", h, level=l) > mean("base", h, level=l) + TIE for l in levels)
    robust = {x: mean("chal", x) - mean("base", x) for x in HIDES if x != h}
    checks = [
        ("a", margin >= BAND - TIE, "margin {:+.4f} (needs >= +{:.2f})".format(margin, BAND)),
        ("b", by_game >= 3, "ahead on {} of {} matches (needs 3)".format(by_game, len(games))),
        ("c", by_level >= 3, "ahead at {} of {} tracker levels (needs 3)".format(
            by_level, len(levels))),
        ("d", all(v >= -TIE for v in robust.values()), "at 15% / 35% hidden: {}".format(
            " / ".join("{:+.4f}".format(robust[x]) for x in sorted(robust)))),
    ]
    return all(ok for _, ok, _ in checks), checks


def table(scores, games, cands):
    lines = []
    for hide in HIDES:
        lines.append("\n**{:.0%} live ball hidden** (spellF1; possessions predicted / reference)"
                     "\n".format(hide))
        lines.append("| Candidate | Match | " + " | ".join(LEVELS) + " | Mean |")
        lines.append("|---" * (len(LEVELS) + 3) + "|")
        for cand in cands:
            for g in games:
                cells = ["{:.3f} ({}/{})".format(*scores[(cand, g, l, hide, "full")])
                         for l in LEVELS]
                m = np.mean([scores[(cand, g, l, hide)] for l in LEVELS])
                lines.append("| {} | {} | {} | {:.4f} |".format(cand, g, " | ".join(cells), m))
            m = np.mean([scores[(cand, g, l, hide)] for g in games for l in LEVELS])
            per = [np.mean([scores[(cand, g, l, hide)] for g in games]) for l in LEVELS]
            lines.append("| **{}** | **mean** | {} | **{:.4f}** |".format(
                cand, " | ".join("{:.4f}".format(v) for v in per), m))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="run on the IDSSE tuning matches; never loads a held-out match, "
                             "writes nothing")
    parser.add_argument("--challenger", nargs=2, metavar=("COMMIT", "LABELER"),
                        help="smoke only: challenger to rehearse with (default: the named one, "
                             "else HEAD and v4)")
    parser.add_argument("--again", action="store_true",
                        help="a deliberate second run (team agreement): appends a new section")
    args = parser.parse_args(argv)

    text = open(REPORT).read() if os.path.isfile(REPORT) else ""
    named = named_challenger(text)
    if args.smoke:
        challenger = tuple(args.challenger) if args.challenger else named or (
            git("rev-parse", "--short", "HEAD"), "v4")
        games = list(bench.IDSSE_TUNING)
    else:
        if args.challenger:
            parser.error("--challenger is for --smoke; the real run uses the named challenger")
        challenger = named
        games = list(bench.IDSSE_HELD_OUT)
    views = [(g, l, h) for h in HIDES for g in games for l in LEVELS]
    print("baseline   {} {}\nchallenger {}\nmatches    {}\nviews      {} per candidate".format(
        BASELINE[0], BASELINE[1], " ".join(challenger) if challenger else "NOT NAMED",
        " ".join(map(str, games)), len(views)))
    if args.dry_run:
        return 0
    if not args.smoke:
        if challenger is None:
            print("refused: no `Challenger: <commit> <labeler>` line in docs/idsse_comparison.md "
                  "(rule 2, 6)")
            return 2
        if MARKER in text and not args.again:
            print("refused: docs/idsse_comparison.md already records a run (rule 6); "
                  "--again only with team agreement")
            return 2
        if dirty():
            print("refused: uncommitted changes to tracked files:\n  " + "\n  ".join(dirty()))
            return 2
    missing = [c for c in (REFERENCE_COMMIT, BASELINE[0]) + ((challenger[0],) if challenger
                                                             else ()) if not has_commit(c)]
    if missing:
        print("refused: commit(s) {} not in this clone; run `git fetch --unshallow`".format(
            " ".join(missing)))
        return 2
    changed = frozen_ok()
    if changed:
        print("refused: reference / degradation code differs from {}: {}".format(
            REFERENCE_COMMIT, ", ".join(changed)))
        return 2

    config.seed_everything()
    head = git("rev-parse", "HEAD")
    tmp = tempfile.mkdtemp(prefix="espadas_idsse_compare_")
    cands = {}
    scores = {}
    try:
        cands["base"] = Candidate(tmp, "base", *BASELINE)
        cands["chal"] = Candidate(tmp, "chal", *challenger)
        if not args.smoke:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(REPORT, "a") as f:
                f.write("\n{} {} (tool at `{}`; baseline `{}`, challenger `{}` {})\n".format(
                    MARKER, stamp, head[:7], cands["base"].full[:7], cands["chal"].full[:7],
                    challenger[1]))
        loaded = dict(bench.available(games, heldout=not args.smoke))
        for k, (g, level, hide) in enumerate(views):
            game = loaded[g]
            gt_codes, tracking, graded = bench.setup(game, hide, level)
            src = os.path.join(tmp, "view.npz")
            save_tracking(src, tracking)
            for tag, cand in cands.items():
                labels = cand.label(src, os.path.join(tmp, tag + ".npy"))
                r = bench.evaluate(gt_codes, labels, graded)
                scores[(tag, g, level, hide)] = r["spell_f1_mean"]
                scores[(tag, g, level, hide, "full")] = (r["spell_f1_mean"], r["n_pred_spells"],
                                                         r["n_gt_spells"])
            print("  {:>2}/{}  match {} {:<6} {:.0%}  base {:.4f}  chal {:.4f}".format(
                k + 1, len(views), g, level, hide, scores[("base", g, level, hide)],
                scores[("chal", g, level, hide)]), flush=True)
    finally:
        for cand in cands.values():
            cand.close()
        shutil.rmtree(tmp, ignore_errors=True)

    adopt, checks = decide(scores, games)
    body = [table(scores, games, ["base", "chal"]), "",
            "**Rule 4** (base = `{}` {}, chal = `{}` {}):".format(
                cands["base"].full[:7], BASELINE[1], cands["chal"].full[:7], challenger[1])]
    body += ["- ({}) {}: {}".format(c, "pass" if ok else "FAIL", why) for c, ok, why in checks]
    body += ["- (e) validator + determinism of the challenger's build on our match: checked "
             "separately, before merging", "",
             "**Result of (a)-(d): {}**".format(
                 "challenger qualifies (subject to (e))" if adopt else "baseline stays")]
    out = "\n".join(body)
    print(out)
    if args.smoke:
        print("\n(smoke: tuning matches only, nothing written)")
        return 0
    with open(REPORT, "a") as f:
        f.write(out + "\n\n**Decision:** _fill in from rule 4 and 7_\n")
    print("\nwrote", os.path.relpath(REPORT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
