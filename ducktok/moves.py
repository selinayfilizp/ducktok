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
    """[t0, t1) windows within the bar for a track's 1-indexed beats."""
    return [((b - 1) * beat, b * beat) for b in track.get("beats", [])]


# Default amplitudes, FK-validated on the Microduck. Override per track.
DEFAULTS = {
    "lift": {"hip_pitch": 0.55, "knee": 0.95, "ankle": 0.15, "amount": 1.0},
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
    for t0, t1 in _beat_windows(track, beat):
        k = bump(tc, t0, t1)
        for joint in profile.hip_roll_joints:
            deltas[joint] += s * k
    return 0.0, 0.0


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


MOVES = {
    "lift": apply_lift,
    "sway": apply_sway,
    "slide": apply_slide,
    "bounce": apply_bounce,
    "head_bob": apply_head_bob,
    "head_sway": apply_head_sway,
}
