"""Compile a choreography spec into a keyframe CSV for motion-imitation RL.

The output format is mjlab's csv_to_npz input: one row per frame at the
requested fps, [base x, y, z, quat x, y, z, w, joints in the profile's
order]. The choreography is defined per bar and evaluated at t mod bar, so
the clip loops seamlessly by construction as long as slides cancel out over
the bar (the compiler warns when they do not).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .moves import DEFAULTS, MOVES, _beat_windows, smoothstep
from .profile import RobotProfile, get_profile


def compile_choreo(
    spec: dict, fps: int = 50
) -> tuple[np.ndarray, RobotProfile, dict]:
    profile = get_profile(spec.get("robot", "microduck"))
    bpm = float(spec["bpm"])
    bars = int(spec.get("bars", 2))
    beat = 60.0 / bpm
    bar = 4.0 * beat
    n_frames = round(bars * bar * fps)

    tracks = spec.get("tracks", [])
    for track in tracks:
        if track["move"] not in MOVES:
            raise KeyError(
                f"Unknown move {track['move']!r}; available: {sorted(MOVES)}"
            )
    joint_tracks = [t for t in tracks if t["move"] != "slide"]
    # Slides are base displacement and must PERSIST across bar boundaries
    # (slide left in bar 2, still left when bar 3 starts), so they are
    # evaluated on global time over per-bar-instanced windows. Every other
    # move is a rest-to-rest envelope inside its bar.
    slide_windows: list[tuple[float, float, float]] = []
    for track in (t for t in tracks if t["move"] == "slide"):
        sign = {"left": 1.0, "right": -1.0}[track["direction"]]
        dist = float(track.get("distance", DEFAULTS["slide"]["distance"]))
        active = track.get("on_bars") or range(1, bars + 1)
        for bar_no in active:
            for w0, w1 in _beat_windows(track, beat):
                slide_windows.append(
                    (sign * dist, (bar_no - 1) * bar + w0, (bar_no - 1) * bar + w1)
                )

    rows = []
    for i in range(n_frames):
        t = i / fps
        tc = t % bar
        bar_idx = int(t // bar) % bars + 1
        deltas = {name: 0.0 for name in profile.joint_order}
        y = sum(d * smoothstep(t, g0, g1) for d, g0, g1 in slide_windows)
        z = profile.stand_height
        for track in joint_tracks:
            on_bars = track.get("on_bars")
            if on_bars is not None and bar_idx not in on_bars:
                continue
            _dy, dz = MOVES[track["move"]](track, tc, beat, profile, deltas)
            z += dz
        joints = []
        for name in profile.joint_order:
            lo, hi = profile.limits[name]
            f = profile.soft_limit_factor
            val = profile.home[name] + deltas[name]
            joints.append(min(max(val, f * lo), f * hi))
        rows.append([0.0, y, z, 0.0, 0.0, 0.0, 1.0] + joints)

    motion = np.asarray(rows)
    first, last = motion[0], motion[-1]
    info = {
        "frames": n_frames,
        "fps": fps,
        "seconds": n_frames / fps,
        "bpm": bpm,
        "bars": bars,
        "beat_s": beat,
        "loop_seam_joint_rad": float(np.abs(first[7:] - last[7:]).max()),
        "loop_seam_base_mm": float(np.abs(first[:3] - last[:3]).max() * 1000),
        "bar_end_y_mm": float(
            motion[min(round(bar * fps), n_frames - 1), 1] * 1000
        ),
    }
    return motion, profile, info


def compile_file(path: str | Path, output: str | Path, fps: int = 50) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    motion, profile, info = compile_choreo(spec, fps=fps)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, motion, delimiter=",", fmt="%.6f")
    info["output"] = str(out)
    info["robot"] = profile.name
    info["name"] = spec.get("name", Path(path).stem)
    return info
