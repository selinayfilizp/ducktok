from pathlib import Path

import numpy as np
import yaml

from ducktok.compile import compile_choreo, compile_file

ROOT = Path(__file__).resolve().parents[1]


def test_toosie_matches_the_trained_reference():
    """The YAML spec must reproduce the exact keyframes the shipped Toosie
    Slide policy was trained on (golden copy from microduck-courier)."""
    spec = yaml.safe_load((ROOT / "choreos" / "toosie_slide.yaml").read_text())
    motion, _, info = compile_choreo(spec, fps=50)
    golden = np.loadtxt(ROOT / "tests" / "golden" / "toosie_slide.csv", delimiter=",")
    assert motion.shape == golden.shape
    assert np.abs(motion - golden).max() < 1e-4
    assert info["loop_seam_base_mm"] < 1.0
    assert abs(info["bar_end_y_mm"]) < 1.0


def test_compile_file_writes_csv(tmp_path):
    out = tmp_path / "toosie.csv"
    info = compile_file(ROOT / "choreos" / "toosie_slide.yaml", out)
    assert out.is_file()
    assert info["frames"] == 293
    data = np.loadtxt(out, delimiter=",")
    assert data.shape == (293, 21)
    assert np.isfinite(data).all()


def test_unbalanced_slides_are_flagged():
    spec = {
        "name": "drifter",
        "robot": "microduck",
        "bpm": 100,
        "bars": 1,
        "tracks": [{"move": "slide", "direction": "left", "beats": [2]}],
    }
    _, _, info = compile_choreo(spec)
    assert abs(info["bar_end_y_mm"]) > 1.0


def test_griddy_compiles_with_clean_loop():
    info = compile_file(ROOT / "choreos" / "griddy.yaml", Path("/tmp/griddy.csv"))
    assert info["frames"] == 304
    assert info["loop_seam_base_mm"] < 1.0
    assert abs(info["bar_end_y_mm"]) < 1.0
