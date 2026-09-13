"""Dance move primitives.

Each move maps (time within the bar, params, robot profile) to joint deltas
and base-pose deltas. Amplitude envelopes are raised-cosine bumps inside a
beat and smoothsteps for travel, so every move starts and ends at rest and
any combination of moves loops seamlessly bar to bar.
"""

from __future__ import annotations

import math

from .profile import RobotProfile


def bump(t: float, t0: float, t1: float) -> float:
    """Raised-cosine 0 -> 1 -> 0 over [t0, t1]."""
    if t <= t0 or t >= t1:
        return 0.0
    x = (t - t0) / (t1 - t0)
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * x))


def smoothstep(t: float, t0: float, t1: float) -> float:
    """Smooth 0 -> 1 over [t0, t1], clamped outside."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


def _beat_windows(track: dict, beat: float) -> list[tuple[float, float]]:
    """Envelope windows within the bar.

    `beats: [1, 3]` gives one per-beat window each; `span: [2, 4]` gives a
    single sustained window from the start of beat 2 to the end of beat 4
    (rise, hold-ish middle, fall), for moves that are held across beats like
    a jump-out wide stance.
    """
    if "span" in track:
        a, b = track["span"]
        return [((a - 1) * beat, b * beat)]
    return [((b - 1) * beat, b * beat) for b in track.get("beats", [])]


# Default amplitudes, FK-validated on the Microduck. Override per track.
DEFAULTS = {
    "lift": {"hip_pitch": 0.55, "knee": 0.95, "ankle": 0.15, "amount": 1.0},
    "kick_back": {"hip_pitch": 1.0, "knee": 1.0, "ankle": 0.1, "amount": 1.0},
    "kick_front": {"hip_pitch": 0.7, "knee": 0.85, "ankle": 0.1, "amount": 1.0},
    "splay": {"amount": 0.20},
    "sway": {"amount": 0.14},
    "slide": {"distance": 0.06},
    "bounce": {"amount": 0.004},
    "head_bob": {"amount": 0.14},
    "head_sway": {"amount": 0.18, "period_bars": 1},
}


def apply_lift(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    p = {**DEFAULTS["lift"], **track}
    side = track["side"]
    s = profile.lift_sign[side] * p["amount"]
    leg = profile.legs[side]
    dz = 0.0
    for t0, t1 in _beat_windows(track, beat):
        k = bump(tc, t0, t1)
        deltas[leg.hip_pitch] += s * p["hip_pitch"] * k
        deltas[leg.knee] += s * p["knee"] * k
        deltas[leg.ankle] += -s * p["ankle"] * k
        # Rise onto the support leg so the planted foot is not pressed
        # through the floor (FK-validated compensation).
        dz += profile.support_rise * k
    return 0.0, dz


def apply_sway(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    p = {**DEFAULTS["sway"], **track}
    s = profile.sway_sign[track["toward"]] * p["amount"]
    dz = 0.0
    for t0, t1 in _beat_windows(track, beat):
        k = bump(tc, t0, t1)
        for joint in profile.hip_roll_joints:
            deltas[joint] += s * k
        # Canting the trunk presses the support-side foot down under a fixed
        # base height (FK: -8.7 mm at amount 0.10). Opt-in rise (meters at
        # full amplitude): default 0 keeps existing specs byte-identical;
        # standalone groove sways should pass rise of about 0.09 * amount.
        dz += p.get("rise", 0.0) * k
    return 0.0, dz


def apply_slide(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    p = {**DEFAULTS["slide"], **track}
    sign = {"left": 1.0, "right": -1.0}[track["direction"]]
    dy = 0.0
    for t0, t1 in _beat_windows(track, beat):
        dy += sign * p["distance"] * smoothstep(tc, t0, t1)
    return dy, 0.0


def apply_bounce(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    p = {**DEFAULTS["bounce"], **track}
    dz = -p["amount"] * 0.5 * (1.0 - math.cos(2.0 * math.pi * tc / beat))
    return 0.0, dz


def apply_head_bob(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    p = {**DEFAULTS["head_bob"], **track}
    for t0, t1 in _beat_windows(track, beat):
        deltas[profile.head_bob_joint] += p["amount"] * bump(tc, t0, t1)
    return 0.0, 0.0


def apply_head_sway(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    p = {**DEFAULTS["head_sway"], **track}
    period = p["period_bars"] * 4.0 * beat
    deltas[profile.head_sway_joint] += p["amount"] * math.sin(
        2.0 * math.pi * tc / period
    )
    return 0.0, 0.0


def apply_kick_back(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    """Heel tap behind the body (the Griddy step): knee flexes hard while the
    hip extends slightly, so the foot rises up AND back instead of up and
    under like a lift. Hip sign FK-validated: on the Microduck's digitigrade
    legs, the hip delta that pulls the foot rearward has the SAME sign as the
    lift's shortening delta (the intuitive opposite sign marches the knee
    forward instead)."""
    p = {**DEFAULTS["kick_back"], **track}
    side = track["side"]
    s = profile.lift_sign[side] * p["amount"]
    leg = profile.legs[side]
    dz = 0.0
    for t0, t1 in _beat_windows(track, beat):
        k = bump(tc, t0, t1)
        deltas[leg.hip_pitch] += s * p["hip_pitch"] * k
        deltas[leg.knee] += s * p["knee"] * k
        deltas[leg.ankle] += -s * p["ankle"] * k
        dz += profile.support_rise * k
    return 0.0, dz


def apply_kick_front(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    """Knee-up front kick: the foot rises AND travels forward. Signs come
    from the same FK sweep that charted kick_back: on the Microduck the
    hip delta OPPOSITE to the lift-shortening sign swings the foot forward
    (48-62 mm lift, 32-51 mm forward in the sweep)."""
    p = {**DEFAULTS["kick_front"], **track}
    side = track["side"]
    s = profile.lift_sign[side] * p["amount"]
    leg = profile.legs[side]
    dz = 0.0
    for t0, t1 in _beat_windows(track, beat):
        k = bump(tc, t0, t1)
        deltas[leg.hip_pitch] += -s * p["hip_pitch"] * k
        deltas[leg.knee] += s * p["knee"] * k
        deltas[leg.ankle] += -s * p["ankle"] * k
        dz += profile.support_rise * k
    return 0.0, dz


def apply_splay(
    track: dict, tc: float, beat: float, profile: RobotProfile, deltas: dict
) -> tuple[float, float]:
    """Wide stance: both legs roll outward (jump-out-and-hold with a span).
    Mirror-signed deltas, unlike sway's same-signed lean; the outward sign
    per side is FK-validated (the first guess pinched the legs inward by
    16 mm per foot instead of splaying them)."""
    p = {**DEFAULTS["splay"], **track}
    left, right = profile.hip_roll_joints
    dz = 0.0
    for t0, t1 in _beat_windows(track, beat):
        k = bump(tc, t0, t1)
        deltas[left] += p["amount"] * k
        deltas[right] += -p["amount"] * k
        # Rolled-out legs shorten the vertical leg projection; rise with the
        # splay or both feet press through the floor (FK: -11.5 mm without; coefficient re-tuned to 0.06 when amount grew to 0.26).
        dz += 0.06 * p["amount"] * k
    return 0.0, dz


MOVES = {
    "lift": apply_lift,
    "kick_back": apply_kick_back,
    "kick_front": apply_kick_front,
    "splay": apply_splay,
    "sway": apply_sway,
    "slide": apply_slide,
    "bounce": apply_bounce,
    "head_bob": apply_head_bob,
    "head_sway": apply_head_sway,
}
