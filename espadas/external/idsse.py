"""Reader for the DFL / Sportec IDSSE open dataset (7 Bundesliga / 2. Bundesliga matches, 2022/23).

Source and licence: tools/fetch_idsse.py (figshare 10.6084/m9.figshare.28196177 v1, CC BY 4.0;
Bassek, Rein, Weber & Memmert, Scientific Data 2025). Files stay outside the repo.

Per match, two of the three DFL XML files are used:
- match information: home / guest team ids (Team Role="home"/"guest") and the pitch size;
- positions ("raw observed", TRACAB Gen5, 25 Hz): one <FrameSet> per object and half
  (GameSection firstHalf / secondHalf), <Frame N=... X=... Y=...> in metres from the centre
  spot. The ball's FrameSet (TeamId "BALL") also carries per frame BallStatus (0 inactive,
  1 active: "inactive when the match is interrupted by the referee") and BallPossession (1 home,
  2 away: "defined by the player in control of the ball"). Referees' FrameSets are skipped.
The events file is not needed: the reference comes from those two flags.

Output, like the Metrica readers: players (n, slots, 2) and ball (n, 2) in metres with the
origin at a corner (x + length / 2, y + width / 2), NaN when an object is not on the pitch;
team_of_slot (0 home, 1 away), one slot per player. The halves are laid end to end with GAP
ungraded frames between them. Plus the reference, which the bench uses instead of events:
labels (H / A / D in the halves, X elsewhere), half_blocks [(period, first, last)] and kickoffs.
Reference rule (fixed, never tuned; chosen 11 Nov before any IDSSE match was scored):
1. BallStatus 0 -> D; else BallPossession 1 -> H, 2 -> A (the provider's flags, `raw` below).
2. **Touches are not changes** (official definition: "a touch without control" is not a change
   of possession). DFL's flag also moves on brief touches: in the tuning matches 27-34% of its
   possessions last under 2 s and a team often "has" the ball for a few frames before it goes
   straight back (44 times in match 12). So a possession spell shorter than TOUCH_FRAMES (1 s)
   that interrupts the other team's live possession, and is not followed by more of the same
   team, is given to the team that had the ball: the ball going back to the other team or going
   dead after it means the touch never became control. A one-touch pass to a team-mate (same
   team next) is kept; a spell right after a dead ball (a restart) is always kept. Effect on the
   tuning matches: 477 / 396 / 389 possessions per 90' -> 379 / 337 / 318 (Metrica: 268, 377);
   0.4-0.7% of frames change, never a D.

Game numbers 11-17 are the matches in match-id order (MATCHES). The split (IDSSE_TUNING /
IDSSE_HELD_OUT) is in espadas/external/__init__.py; a held-out match is refused before any file
is opened unless heldout=True. Parsing a positions file takes a minute or two, so the arrays
are cached next to the data (idsse_dir()/cache, outside the repo); the cache is keyed on the
file's size and modification time and never changes the result. Deterministic.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import check_split, idsse_dir

FPS = 25
GAP = 250                      # ungraded frames between the halves
TOUCH_FRAMES = 25              # 1 s: shorter possessions that do not hold are touches (rule 2)
CACHE_VERSION = 1

MATCHES = {                    # game number -> (competition, match id)
    11: ("DFL-COM-000001", "DFL-MAT-J03WMX"),
    12: ("DFL-COM-000001", "DFL-MAT-J03WN1"),
    13: ("DFL-COM-000002", "DFL-MAT-J03WOH"),
    14: ("DFL-COM-000002", "DFL-MAT-J03WOY"),
    15: ("DFL-COM-000002", "DFL-MAT-J03WPY"),
    16: ("DFL-COM-000002", "DFL-MAT-J03WQQ"),
    17: ("DFL-COM-000002", "DFL-MAT-J03WR9"),
}
SECTIONS = ("firstHalf", "secondHalf")


def paths(game, heldout=False):
    """(match information, positions) file paths; refuses a held-out match first."""
    check_split([game], heldout)
    competition, match = MATCHES[game]
    root = idsse_dir()
    return (os.path.join(root, "DFL_02_01_matchinformation_{}_{}.xml".format(competition, match)),
            os.path.join(root, "DFL_04_03_positions_raw_observed_{}_{}.xml".format(
                competition, match)))


def read_match_information(path):
    """dict(home, away team ids, pitch (length, width), title)."""
    root = ET.parse(path).getroot()
    teams = {t.get("Role"): t.get("TeamId") for t in root.iter("Team")}
    env = root.find(".//Environment")
    general = root.find(".//General")
    if "home" not in teams or "guest" not in teams:
        raise ValueError("{}: expected a home and a guest team".format(path))
    return {"home": teams["home"], "away": teams["guest"],
            "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
            "title": general.get("MatchTitle") if general is not None else ""}


def parse_positions(path, home, away):
    """{section: {"ball": (N, x, y, status, possession), "players": {person: (team, N, x, y)}}}.

    Streams the file (iterparse), so memory stays at the size of the arrays."""
    out = {}
    current = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            team = elem.get("TeamId")
            kind = "ball" if team and team.upper() == "BALL" else \
                "player" if team in (home, away) else None
            current = {"kind": kind, "team": 0 if team == home else 1,
                       "section": elem.get("GameSection"), "person": elem.get("PersonId"),
                       "rows": []}
        elif event == "end" and elem.tag == "Frame":
            if current is not None and current["kind"] is not None:
                if current["kind"] == "ball":
                    current["rows"].append((int(elem.get("N")), float(elem.get("X")),
                                            float(elem.get("Y")),
                                            int(elem.get("BallStatus", -1)),
                                            int(elem.get("BallPossession", 0))))
                else:
                    current["rows"].append((int(elem.get("N")), float(elem.get("X")),
                                            float(elem.get("Y"))))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            if current is not None and current["kind"] is not None:
                section = out.setdefault(current["section"], {"ball": None, "players": {}})
                rows = np.array(current["rows"], dtype=np.float64)
                if current["kind"] == "ball":
                    section["ball"] = rows
                else:
                    section["players"][current["person"]] = (current["team"], rows)
            current = None
            elem.clear()
    return out


def assemble(parsed, pitch):
    """Arrays for the whole match from parse_positions() output (see module docstring)."""
    sections = [s for s in SECTIONS if s in parsed]
    if not sections or any(parsed[s]["ball"] is None for s in sections):
        raise ValueError("positions: missing halves or ball data")
    persons = sorted({p for s in sections for p in parsed[s]["players"]})
    team_of = {}
    for s in sections:
        for person, (team, _) in parsed[s]["players"].items():
            team_of[person] = team
    persons.sort(key=lambda p: (team_of[p], p))          # home slots first, then away
    slot_of = {p: k for k, p in enumerate(persons)}

    spans, offset = [], 0
    for s in sections:
        frames = parsed[s]["ball"][:, 0].astype(np.int64)
        first, last = int(frames.min()), int(frames.max())
        spans.append((s, first, last, offset))
        offset += (last - first + 1) + GAP
    n = offset - GAP

    players = np.full((n, len(persons), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    labels = np.full(n, "X", dtype="<U1")
    half_blocks = []
    centre = np.array(pitch, dtype=np.float64) / 2.0
    for period, (s, first, last, start) in enumerate(spans, 1):
        b = parsed[s]["ball"]
        idx = b[:, 0].astype(np.int64) - first + start
        ball[idx] = b[:, 1:3] + centre
        status, possession = b[:, 3].astype(np.int64), b[:, 4].astype(np.int64)
        seg = np.full(len(b), "D", dtype="<U1")
        seg[(status == 1) & (possession == 1)] = "H"
        seg[(status == 1) & (possession == 2)] = "A"
        labels[start:start + last - first + 1] = "D"     # frames without a ball row: dead
        labels[idx] = seg
        for person, (_, rows) in parsed[s]["players"].items():
            k = rows[:, 0].astype(np.int64) - first + start
            ok = (k >= start) & (k <= start + last - first)
            players[k[ok], slot_of[person]] = rows[ok, 1:3] + centre
        half_blocks.append((period, start, start + last - first))
    team_of_slot = np.array([team_of[p] for p in persons], dtype=np.int8)
    kickoffs = tuple(first for _, first, _ in half_blocks)
    return {"players": players, "ball": ball, "team_of_slot": team_of_slot, "labels": labels,
            "half_blocks": half_blocks, "kickoffs": kickoffs}


def spells(labels):
    """[(label, start, stop)] runs of equal labels."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(labels)]))
    return [(str(labels[a]), int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def official_labels(raw, touch_frames=TOUCH_FRAMES):
    """Rule 2 of the module docstring applied to the provider's labels (copy). Deterministic."""
    out = np.asarray(raw, dtype="<U1").copy()
    runs = spells(out)
    owner = None
    for i, (label, start, stop) in enumerate(runs):
        if label not in "HA":
            continue
        after_live = i > 0 and runs[i - 1][0] in ("H", "A")
        holds = i + 1 < len(runs) and runs[i + 1][0] == label
        if (after_live and owner is not None and label != owner
                and stop - start < touch_frames and not holds):
            out[start:stop] = owner           # a touch: the ball stays with the owner
        else:
            owner = label
    return out


def _cache_path(positions):
    st = os.stat(positions)
    name = "{}.v{}.{}.{}.npz".format(os.path.basename(positions), CACHE_VERSION, st.st_size,
                                     int(st.st_mtime))
    return os.path.join(idsse_dir(), "cache", name)


def load_arrays(game, heldout=False, use_cache=True):
    info_path, positions = paths(game, heldout)
    info = read_match_information(info_path)
    cache = _cache_path(positions)
    if use_cache and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["half_blocks"] = [tuple(int(v) for v in row) for row in arrays["half_blocks"]]
        arrays["kickoffs"] = tuple(int(v) for v in arrays["kickoffs"])
        return info, arrays
    arrays = assemble(parse_positions(positions, info["home"], info["away"]), info["pitch"])
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez(tmp, **dict(arrays, half_blocks=np.array(arrays["half_blocks"]),
                             kickoffs=np.array(arrays["kickoffs"])))
        os.replace(tmp, cache)
    return info, arrays


def load_game(game, heldout=False):
    """The match as the bench needs it, with its reference built in (no events needed)."""
    info, a = load_arrays(game, heldout)
    tracking = {"players": a["players"], "ball": a["ball"], "team_of_slot": a["team_of_slot"]}
    return {"name": "idsse{}".format(game), "tracking": tracking, "events": None,
            "reference": (official_labels(a["labels"]), a["half_blocks"], a["kickoffs"]),
            "raw_labels": a["labels"],
            "n_frames": len(a["ball"]), "fps": FPS, "pitch": info["pitch"],
            "title": info["title"]}
