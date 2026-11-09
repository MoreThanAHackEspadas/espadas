"""Reader for Metrica Sample Games 1 and 2 (older CSV format).

Tracking: one CSV per team, three header rows (team, jersey, column names), then
`Period, Frame, Time [s], Player<n> x, y, ..., Ball x, y` with 0-1 coordinates and NaN when
missing. Frames are 1-based and the two files share the frame column. Each player is a stable
column, so each player gets one slot (unlike our match's tracklet slots).

Events: one CSV, `Team, Type, Subtype, Period, Start Frame, ..., End Frame, ...`.

Output matches what the rest of the pipeline expects: players (n, slots, 2) and ball (n, 2) in
metres, team_of_slot (0 home, 1 away), 0-based frames, plus normalised events.

Game 2 is held out (checklist 4.4): every path goes through game_dir(), which refuses it unless
the caller passes heldout=True, before any file is opened.
"""

import os

import numpy as np
import pandas as pd

from . import check_split, metrica_dir

PITCH = (105.0, 68.0)
FPS = 25


def game_dir(game, heldout=False):
    check_split([game], heldout)
    return os.path.join(metrica_dir(), "data", "Sample_Game_{}".format(game))


def _read_team(path):
    """(frames 1-based, xy (n, players, 2) normalised, ball (n, 2) normalised)."""
    names = pd.read_csv(path, skiprows=2, nrows=0).columns.tolist()
    table = pd.read_csv(path, skiprows=3, header=None).to_numpy(np.float64)
    frames = table[:, 1].astype(np.int64)
    xy = table[:, 3:]
    if xy.shape[1] % 2 or not str(names[-2]).startswith("Ball"):
        raise ValueError("{}: unexpected column layout".format(path))
    pairs = xy.reshape(len(table), -1, 2)
    return frames, pairs[:, :-1], pairs[:, -1]


def load_tracking(game, heldout=False):
    folder = game_dir(game, heldout)
    stem = "Sample_Game_{}_RawTrackingData_{}_Team.csv"
    frames_h, home, ball = _read_team(os.path.join(folder, stem.format(game, "Home")))
    frames_a, away, _ = _read_team(os.path.join(folder, stem.format(game, "Away")))
    if not np.array_equal(frames_h, frames_a):
        raise ValueError("game {}: home and away frame columns differ".format(game))
    if not np.array_equal(frames_h, np.arange(1, len(frames_h) + 1)):
        raise ValueError("game {}: frames are not 1..N contiguous".format(game))

    scale = np.array(PITCH)
    players = np.concatenate([home, away], axis=1) * scale
    team_of_slot = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    return {"players": players, "ball": ball * scale, "team_of_slot": team_of_slot}


def load_events(game, heldout=False):
    path = os.path.join(game_dir(game, heldout), "Sample_Game_{}_RawEventsData.csv".format(game))
    table = pd.read_csv(path)
    events = []
    for order, row in enumerate(table.itertuples(index=False)):
        team = {"Home": 0, "Away": 1}.get(row.Team)
        end = int(row[6])
        events.append({
            "order": order,
            "team": team,
            "type": str(row.Type).strip().upper(),
            "subtype": "" if pd.isna(row.Subtype) else str(row.Subtype).strip().upper(),
            "period": int(row.Period),
            "start": int(row[4]) - 1,
            "end": end - 1 if end > 0 else None,
        })
    return events


def load_game(game, heldout=False):
    """The game as the bench needs it. A held-out game needs heldout=True (checklist 4.4)."""
    tracking = load_tracking(game, heldout)
    return {"name": "metrica{}".format(game), "tracking": tracking,
            "events": load_events(game, heldout), "n_frames": len(tracking["ball"]),
            "fps": FPS, "pitch": PITCH}
