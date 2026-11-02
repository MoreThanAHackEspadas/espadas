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
    """'x,y,speed' -> (x, y, speed) floats, NaN when missing."""
    if not group or group[0] == "N":
        return np.nan, np.nan, np.nan
    fields = group.split(",")
    speed = float(fields[2]) if len(fields) > 2 else np.nan
    return float(fields[0]), float(fields[1]), speed


def parse_raw(raw_path, n_slots, n_frames=config.N_FRAMES, pitch=config.PITCH):
    """Players (n_frames, n_slots, 2) and ball (n_frames, 2) in metres, float64 (matches the baseline exactly), NaN if missing.

    Also the speed the tracker reports for each object, in m/s: player_speed (n_frames, n_slots)
    and ball_speed (n_frames,), float32. Frames the file does not mention stay NaN. The tracker
    sometimes reports a position with no speed, so a speed can be NaN where the position is not;
    prep.ball_speed() derives speed from positions instead.
    """
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float64)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    player_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)

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
                x, y, s = _parse_group(groups[slot])
                players[frame, slot, 0] = x * length
                players[frame, slot, 1] = y * width
                player_speed[frame, slot] = s
            bx, by, bs = _parse_group(parts[2])
            ball[frame, 0] = bx * length
            ball[frame, 1] = by * width
            ball_speed[frame] = bs
    return players, ball, player_speed, ball_speed


# Bump whenever parse_raw's output changes, so an old cache is never read back.
CACHE_VERSION = 2
CACHED = ("players", "ball", "player_speed", "ball_speed")


def _cache_path(raw_path):
    stat = os.stat(raw_path)
    key = "{}|{}|{}|{}|v{}".format(os.path.basename(raw_path), stat.st_size, int(stat.st_mtime),
                                   config.PITCH, CACHE_VERSION)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(config.CACHE_DIR, "atd_{}.npz".format(digest))


def load_tracking(data_dir=None, use_cache=True):
    """dict(players, ball, player_speed, ball_speed, team_of_slot) for the tactical source."""
    meta_path, raw_path = config.require(["atd_meta", "atd_raw"], data_dir)
    team_of_slot = team_of_each_track(meta_path)

    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            tracking = {name: z[name] for name in CACHED}
    else:
        tracking = dict(zip(CACHED, parse_raw(raw_path, len(team_of_slot))))
        if use_cache:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            np.savez(cache, **tracking)
    tracking["team_of_slot"] = team_of_slot
    return tracking
