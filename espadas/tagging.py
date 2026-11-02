"""Step 1 input: read the Smart Tagging XML onto video frames.

SportsCode XML (see the organisers' starter/load_smart_tagging.py, MIT): one <instance> per phase
or set piece, with <start>/<end> in seconds from the first frame of the tactical video, so
video_frame = round(seconds * fps). The organisers' time-base check confirms this on the file.

The export mixes case (codes upper, tag groups and values title case), so everything is
normalised here: codes upper case, tag groups and values lower case.

The tags are automatic and approximate, not ground truth. The Team tag can be N/A (every
throw-in), and a defensive phase names the defending side, not the possessor. Mapping codes to
possession evidence is a modelling choice and belongs to the evidence step, not this loader.
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import config

HOME, AWAY = 0, 1

# Club names as they appear in the Team tag (DATA.md: 'Barcelona', 'Atletic Club'), lower case.
# 'athletic club' is accepted too in case the export ever corrects the spelling.
TEAM_NAMES = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}

SET_PIECES = "SET PIECES"


def parse(path, fps=config.FPS):
    """[{code, start_frame, end_frame, tags}] in file order. end_frame is exclusive."""
    events = []
    for instance in ET.parse(path).iter("instance"):
        tags = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip().lower()
            if group:
                tags[group] = (label.findtext("text") or "").strip().lower()  # last one wins
        events.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(instance.findtext("start")) * fps)),
            "end_frame": int(round(float(instance.findtext("end")) * fps)),
            "tags": tags,
        })
    return events


def load_events(data_dir=None):
    """Smart Tagging events for the tactical source."""
    (path,) = config.require(["smart_tagging"], data_dir)
    return parse(path)


def team_of(event):
    """HOME or AWAY from the Team tag; None for N/A, a missing tag or an unknown name."""
    return TEAM_NAMES.get(event["tags"].get("team", ""))


def unknown_team_names(events):
    """Team tag values that are neither N/A nor a known club. Should be empty."""
    names = {e["tags"].get("team") for e in events if "team" in e["tags"]}
    return sorted(n for n in names if n not in TEAM_NAMES and n != "n/a")


def interval_mask(events, keep, n_frames=config.N_FRAMES):
    """bool per frame: True inside [start_frame, end_frame) of every event where keep(event)."""
    mask = np.zeros(n_frames, dtype=bool)
    for event in events:
        if keep(event):
            start, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
            if start < end:
                mask[start:end] = True
    return mask


def set_piece_mask(events, n_frames=config.N_FRAMES):
    """Frames inside any SET PIECES window. A window around a restart, not a dead-ball boundary."""
    return interval_mask(events, lambda e: e["code"] == SET_PIECES, n_frames)
