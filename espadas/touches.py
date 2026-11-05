"""Stage 2: touches and control from tracking (checklist Phase 6).

    near, slot, dist = touches.nearest_slot(players, ball)
    t = touches.detect(touches.kink(ball, fps, touches.KINK_HALF), near, slot, team_of_slot)
    ok = touches.confirmed(t, dist)             # touch starts / continues control by its team
    lo = touches.pulses(n, t, ok, 45.0, 10.0)   # per-frame evidence for the decoder (v3)

**Touch (6.1).** The ball's velocity changes sharply (a kick, a trap, a deflection) while a player
is within `touch_reach` of it. Velocity is the ball's position change over 2 frames; the kink is
how much the velocity differs KINK_HALF frames before and after. Local peaks above
`kink_threshold` m/s, at least MIN_SEPARATION frames apart, are touches. On Metrica 1 and 3 with
25% of live ball hidden this finds ~70% of the event-data touches, and puts ~90% of them on the
right team; about a third of detected touches have no event (mostly dribble touches, which the
Metrica events do not record as such).

**Team (6.2).** The team of the nearest tracked player at the touch frame. Slots are used only
within a few seconds (pass chains, holds), so a slot changing player at a substitution or a
tracking swap does not matter.

**Control (6.3).** A touch alone does not change possession (official rules: a deflection, a
block or a won header is not control). A team's touch is *confirmed* when:
- its next touch is by the same team within `chain_gap` frames (a pass to a teammate, or a
  player touching the ball again while dribbling), or
- the same player stays within `hold_reach` of the ball for `hold_frames` frames after it.
**Into the decoder.** Touches become per-frame evidence (`pulses`): a confirmed touch adds a big
pulse for its team, any other touch a small one. v3 adds these to the Phase 5 team log-odds and
runs v2's CUSUM, so a confirmed touch by the other team can switch possession at once (dated back
to when the evidence against the owner started), while during ball gaps the slower clues (team
shape, pressing) still drive switches. Using touches *alone* was tried and was worse on Metrica 3:
when the ball is hidden at the moment it changes hands, no touch is seen and the change is missed.

Every setting is tuned on the external games only (tools/tune.py), never on our match.
"""

import numpy as np

HOME, AWAY = 0, 1
MIN_SEPARATION = 5          # frames between two touches
KINK_HALF = 2               # default frames each side for the velocity change
KINK_HALVES = (2, 4, 6, 10)  # the widths v3 precomputes and may tune over (6.6: blurred balls)


def ball_velocity(ball, fps):
    """m/s per frame (x, y), central difference; NaN where a neighbour is missing."""
    v = np.full(ball.shape, np.nan)
    if len(ball) > 2:
        v[1:-1] = (ball[2:] - ball[:-2]) * fps / 2
    return v


def kink(ball, fps, half):
    """|velocity after - velocity before| over +/- half frames (m/s); 0 where unknown."""
    v = ball_velocity(ball, fps)
    out = np.zeros(len(ball))
    if len(ball) > 2 * half:
        d = v[2 * half:] - v[:len(ball) - 2 * half]
        out[half:len(ball) - half] = np.nan_to_num(np.hypot(d[:, 0], d[:, 1]))
    return out


def nearest_slot(players, ball):
    """(distance, slot) of the nearest tracked player per frame; inf / -1 where unknown."""
    diff = players - ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    slot = np.argmin(dist, axis=1)
    near = dist[np.arange(len(dist)), slot]
    slot = np.where(np.isfinite(near), slot, -1)
    return near, slot, dist


def detect(k, near, slot, team_of_slot, kink_threshold=3.0, touch_reach=2.0):
    """Touches as a dict of arrays: frame, slot, team, strength (kink m/s). Time order.

    k = kink(ball, fps, KINK_HALF); near, slot from nearest_slot(players, ball)."""
    peak = np.zeros(len(k), dtype=bool)
    if len(k) > 2:
        peak[1:-1] = (k[1:-1] > kink_threshold) & (k[1:-1] >= k[:-2]) & (k[1:-1] > k[2:])
    candidates = np.flatnonzero(peak & (near <= touch_reach))

    keep = []
    for i in candidates:                     # keep the strongest peak within MIN_SEPARATION
        if keep and i - keep[-1] < MIN_SEPARATION:
            if k[i] > k[keep[-1]]:
                keep[-1] = int(i)
        else:
            keep.append(int(i))
    frames = np.array(keep, dtype=np.int64)
    slots = slot[frames]
    return {"frame": frames, "slot": slots,
            "team": np.asarray(team_of_slot, dtype=np.int8)[slots],
            "strength": k[frames]}


def carrier(near, slot, present, team_of_slot, reach=2.5, min_frames=6):
    """Touches from changes of *who has the ball at their feet* (checklist 6.7), not from the
    ball's path.

    A frame's carrier is the nearest tracked player if the ball is tracked and that player is
    within `reach`. A carrier spell is a run of frames with the same carrier, at least
    `min_frames` long; a touch is the first frame of a spell whose carrier differs from the
    previous spell's. strength = the spell's length in frames.

    Why: the kink detector needs the ball's path to bend at a kick. The Metrica ball is drawn as
    straight lines, so every kick is a sharp corner, but our tracker's ball rarely bends: with
    the kink detector we found ~290 touches per 90' on our match vs ~1,800 on the blurred bench.
    Carrier changes need no bend; on Metrica 1 (reach 2.5 m, 6 frames) they give 1,200-1,360
    touches per 90' across the four tracker levels and 1,295 on our match. Their team is
    noisier under player position error (right ~70% on light/medium, ~50% on heavy/smooth), so
    the decoder's confirmation and switching costs have to absorb that.
    """
    c = np.where(np.asarray(present, dtype=bool) & (near <= reach), slot, -1)
    if len(c) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return {"frame": empty, "slot": empty, "team": empty.astype(np.int8),
                "strength": np.zeros(0)}
    edges = np.flatnonzero(np.diff(c) != 0) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(c)]))
    frames, slots, lengths = [], [], []
    last = -1
    for a, b in zip(starts, stops):
        who = int(c[a])
        if who < 0 or b - a < min_frames:
            continue
        if who != last:
            frames.append(int(a))
            slots.append(who)
            lengths.append(float(b - a))
        last = who
    frames = np.array(frames, dtype=np.int64)
    slots = np.array(slots, dtype=np.int64)
    return {"frame": frames, "slot": slots,
            "team": np.asarray(team_of_slot, dtype=np.int8)[slots],
            "strength": np.array(lengths)}


def merge(a, b):
    """Touches from two detectors in time order; a touch of `b` within MIN_SEPARATION frames of
    one of `a` is dropped (same touch seen twice)."""
    keep = np.ones(len(b["frame"]), dtype=bool)
    if len(a["frame"]):
        j = np.searchsorted(a["frame"], b["frame"])
        for i, (f, x) in enumerate(zip(b["frame"], j)):
            for q in (x - 1, x):
                if 0 <= q < len(a["frame"]) and abs(int(a["frame"][q]) - int(f)) < MIN_SEPARATION:
                    keep[i] = False
    out = {k: np.concatenate([a[k], b[k][keep]]) for k in ("frame", "slot", "team", "strength")}
    order = np.argsort(out["frame"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def pulses(n, t, ok, confirm_weight, touch_weight):
    """Home-positive evidence per frame from touches: confirm_weight at confirmed touches,
    touch_weight at the others, signed by the touching team."""
    out = np.zeros(n)
    sign = np.where(t["team"] == HOME, 1.0, -1.0)
    np.add.at(out, t["frame"], sign * np.where(ok, confirm_weight, touch_weight))
    return out


def held(frame, slot, dist, hold_reach, hold_frames):
    """True if `slot` stays within hold_reach of the ball for hold_frames frames after `frame`
    (frames where the ball is not tracked count as neither)."""
    seg = dist[frame + 1:frame + 1 + int(hold_frames), slot]
    seg = seg[np.isfinite(seg)]
    return len(seg) >= hold_frames // 2 and bool(np.all(seg <= hold_reach))


def confirmed(t, dist, chain_gap=100, hold_reach=1.5, hold_frames=38):
    """bool per touch: this touch starts or continues controlled possession by its team."""
    n = len(t["frame"])
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        f, team = int(t["frame"][i]), int(t["team"][i])
        if i + 1 < n and t["team"][i + 1] == team and t["frame"][i + 1] - f <= chain_gap:
            out[i] = True
        elif hold_frames > 0 and held(f, int(t["slot"][i]), dist, hold_reach, hold_frames):
            out[i] = True
    return out


def from_features(f, p):
    """Touches and their confirmed flags from a labeler's features (v3.features) and settings.
    p["touch_mode"]: "kink" (ball path bends, Phase 6), "carrier" (change of who has the ball at
    their feet, 6.7) or "both". Shared by v3 and v4."""
    mode = p.get("touch_mode", "kink")
    kinked = detect(f["kinks"][int(p["kink_half"])], f["near"], f["slot"], f["team_of_slot"],
                    p["kink_threshold"], p["touch_reach"])
    if mode == "kink":
        t = kinked
    else:
        carry = carrier(f["near"], f["slot"], f["present"], f["team_of_slot"],
                        p.get("carrier_reach", 2.5), int(p.get("carrier_frames", 6)))
        t = carry if mode == "carrier" else merge(kinked, carry)
    ok = confirmed(t, f["dist"], int(p["chain_gap"]), p["hold_reach"], int(p["hold_frames"]))
    return t, ok
