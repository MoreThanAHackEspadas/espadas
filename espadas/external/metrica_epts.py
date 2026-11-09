"""Reader for Metrica Sample Game 3 (FIFA EPTS: metadata XML + raw tracking TXT + events JSON).

Same line syntax as our match (`frame:x,y;x,y;...:x,y`, 1-based frames, 0-1 coordinates, NaN
for missing), but **not** the same player layout, so atd.parse_raw() must not be used on it:

- Our match: group k on every line is the k-th <Player> (a fixed tracklet slot).
- Game 3: 22 groups per line, and which player each group is changes at substitutions. The
  metadata has one <DataFormatSpecification> per frame range, listing the PlayerChannelRef ids
  in group order; <PlayerChannel> maps each channel id to a playerId.

Here every <Player> gets one stable slot (35 for Game 3), team_of_slot comes from its teamId,
and home is the first <Team> (also the Score idLocalTeam). Output arrays match the rest of the
pipeline: players (n, slots, 2) and ball (n, 2) in metres, 0-based frames.

Like metrica_csv, game_dir() refuses a held-out game (checklist 4.4) unless the caller passes
heldout=True, before any file is opened.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import check_split, metrica_dir

FPS = 25
GAME = 3


def game_dir(game=GAME, heldout=False):
    check_split([game], heldout)
    return os.path.join(metrica_dir(), "data", "Sample_Game_{}".format(game))


def _paths(game=GAME, heldout=False):
    folder, stem = game_dir(game, heldout), "Sample_Game_{}".format(game)
    return (os.path.join(folder, stem + "_metadata.xml"),
            os.path.join(folder, stem + "_tracking.txt"),
            os.path.join(folder, stem + "_events.json"))


def read_metadata(path):
    """dict(pitch, team_ids [home, away], player_ids, team_of_slot, specs).

    specs: [(start_frame, end_frame, [slot of group 0, slot of group 1, ...])], 1-based inclusive.
    """
    root = ET.parse(path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(path, len(team_ids)))
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    slot_of = {pid: k for k, (pid, _) in enumerate(players)}
    team_of_slot = np.array([0 if tid == team_ids[0] else 1 for _, tid in players], dtype=np.int8)

    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        xs = [r for r in refs if r.endswith("_x")]
        slots = [slot_of[channel_player[r]] for r in xs]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))

    size = root.find(".//FieldSize")
    pitch = (float(size.findtext("Width")), float(size.findtext("Height")))
    return {"pitch": pitch, "team_ids": team_ids, "player_ids": [p for p, _ in players],
            "team_of_slot": team_of_slot, "specs": specs}


def read_tracking(path, meta):
    n_frames = max(end for _, end, _ in meta["specs"])
    n_slots = len(meta["team_of_slot"])
    players = np.full((n_frames, n_slots, 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)

    specs = sorted(meta["specs"])
    k, (start, end, slots) = 0, specs[0]
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            while frame > end:
                k += 1
                start, end, slots = specs[k]
            if frame < start:
                raise ValueError("frame {} not covered by any DataFormatSpecification".format(frame))
            groups = parts[1].split(";")
            if len(groups) != len(slots):
                raise ValueError("frame {}: {} groups, spec has {}".format(
                    frame, len(groups), len(slots)))
            for slot, group in zip(slots, groups):
                x, y = group.split(",")[:2]
                players[frame - 1, slot] = (float(x), float(y))
            bx, by = parts[2].split(",")[:2]
            ball[frame - 1] = (float(bx), float(by))

    scale = np.array(meta["pitch"])
    return players * scale, ball * scale


def _subtype(value):
    if not value:
        return ""
    if isinstance(value, list):
        return "-".join(str(v["name"]).strip().upper() for v in value)
    return str(value["name"]).strip().upper()


def read_events(path, team_ids):
    with open(path) as handle:
        data = json.load(handle)["data"]
    events = []
    for order, e in enumerate(data):
        team = team_ids.index(e["team"]["id"]) if e.get("team") else None
        end = (e.get("end") or {}).get("frame")
        events.append({
            "order": order,
            "team": team,
            "type": str(e["type"]["name"]).strip().upper(),
            "subtype": _subtype(e.get("subtypes")),
            "period": int(e["period"]),
            "start": int(e["start"]["frame"]) - 1,
            "end": int(end) - 1 if end else None,
        })
    return events


def load_game(game=GAME, heldout=False):
    """The game as the bench needs it. A held-out game needs heldout=True (checklist 4.4)."""
    meta_path, tracking_path, events_path = _paths(game, heldout)
    meta = read_metadata(meta_path)
    players, ball = read_tracking(tracking_path, meta)
    tracking = {"players": players, "ball": ball, "team_of_slot": meta["team_of_slot"]}
    return {"name": "metrica{}".format(game), "tracking": tracking,
            "events": read_events(events_path, meta["team_ids"]), "n_frames": len(ball),
            "fps": FPS, "pitch": meta["pitch"]}
