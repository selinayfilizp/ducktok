"""ducktok CLI: compile a choreography YAML into a keyframe CSV.

    python -m ducktok choreos/toosie_slide.yaml -o toosie_slide.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .compile import compile_file


def main() -> None:
    parser = argparse.ArgumentParser(prog="ducktok", description=__doc__)
    parser.add_argument("choreo", type=Path, help="Choreography YAML file")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    output = args.output or args.choreo.with_suffix(".csv")
    info = compile_file(args.choreo, output, fps=args.fps)
    print(
        f"{info['name']} ({info['robot']}): {info['frames']} frames at "
        f"{info['fps']} fps, {info['bars']} bars of {info['bpm']} BPM "
        f"({info['seconds']:.2f} s) -> {info['output']}"
    )
    print(
        f"loop seam: {info['loop_seam_joint_rad']:.4f} rad joints, "
        f"{info['loop_seam_base_mm']:.1f} mm base; "
        f"bar-end drift {info['bar_end_y_mm']:.1f} mm"
    )
    if abs(info["bar_end_y_mm"]) > 1.0:
        print(
            "WARNING: slides do not cancel over the bar; the dance will "
            "travel and the loop seam will jump. Balance the slide tracks "
            "unless that is intentional."
        )


if __name__ == "__main__":
    main()
