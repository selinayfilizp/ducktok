"""ducktok: beat-grid choreography compiler for motion-imitation RL.

Write a dance as a small YAML of beat-indexed moves, compile it against a
robot profile with FK-validated sign conventions, and train it with any
BeyondMimic-style tracking task (reference pipeline: the Microduck).
"""

from .compile import compile_choreo, compile_file
from .profile import PROFILES, RobotProfile, get_profile

__all__ = [
    "compile_choreo",
    "compile_file",
    "get_profile",
    "PROFILES",
    "RobotProfile",
]
__version__ = "0.1.0"
