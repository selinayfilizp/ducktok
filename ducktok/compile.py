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

from .moves import MOVES
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

    rows = []
    for i in range(n_frames):
        t = i / fps
        tc = t % bar
        deltas = {name: 0.0 for name in profile.joint_order}
        y = 0.0
        z = profile.stand_height
        for track in spec.get("tracks", []):
            move = MOVES.get(track["move"])
            if move is None:
                raise KeyError(
                    f"Unknown move {track['move']!r}; available: {sorted(MOVES)}"
                )
            dy, dz = move(track, tc, beat, profile, deltas)
            y += dy
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
