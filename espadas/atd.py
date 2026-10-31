"""Step 1 input: read the FIFA/EPTS tracking (ATD) into arrays.

Format (see the organisers' starter/baseline.py, MIT): one line per frame,

    <frame>:<x,y,speed>;<x,y,speed>;...:<x,y,speed>

players first (one group per track slot, 46 slots), then the ball. x and y are normalised 0-1,
"NaN" marks a missing observation, and <frame> counts video frames from 1.

The parse is slow (a couple of minutes), so the result is cached as .npz under CACHE_DIR keyed by
the raw file's size and mtime. The cache is an optimisation only: the output is identical with
or without it.
"""

import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import config

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """int8 array over track slots: 0 home, 1 away. First <Team> in the XML is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_group(group):
    """'x,y,speed' -> (x, y) floats, NaN when missing."""
    if not group or group[0] == "N":
        return np.nan, np.nan
    x, y = group.split(",")[:2]
    return float(x), float(y)


def parse_raw(raw_path, n_slots, n_frames=config.N_FRAMES, pitch=config.PITCH):
    """Players (n_frames, n_slots, 2) and ball (n_frames, 2) in metres, float64 (matches the baseline exactly), NaN if missing.

    Frames the file does not mention stay NaN.
    """
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float64)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # 1-based in the file, 0-based in the submission
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            for slot in range(min(n_slots, len(groups))):
                x, y = _parse_group(groups[slot])
                players[frame, slot, 0] = x * length
                players[frame, slot, 1] = y * width
            bx, by = _parse_group(parts[2])
            ball[frame, 0] = bx * length
            ball[frame, 1] = by * width
    return players, ball


def _cache_path(raw_path):
    stat = os.stat(raw_path)
    key = "{}|{}|{}|{}".format(os.path.basename(raw_path), stat.st_size, int(stat.st_mtime),
                               config.PITCH)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(config.CACHE_DIR, "atd_{}.npz".format(digest))


def load_tracking(data_dir=None, use_cache=True):
    """dict(players, ball, team_of_slot) for the tactical source."""
    meta_path, raw_path = config.require(["atd_meta", "atd_raw"], data_dir)
    team_of_slot = team_of_each_track(meta_path)

    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            return {"players": z["players"], "ball": z["ball"], "team_of_slot": team_of_slot}

    players, ball = parse_raw(raw_path, len(team_of_slot))
    if use_cache:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players, ball=ball)
    return {"players": players, "ball": ball, "team_of_slot": team_of_slot}
