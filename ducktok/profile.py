"""Robot profiles: everything a choreography compiler must know about a body.

A profile encodes the joint order, the neutral standing pose, joint limits,
and, crucially, the SIGN CONVENTIONS for each dance primitive. Signs are the
part everyone gets wrong on a new robot (the first Microduck draft drove a
foot 17.5 mm through the floor); they are validated once per robot with a
forward-kinematics replay and then live here so no choreography author ever
thinks about them again.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegJoints:
    hip_pitch: str
    knee: str
    ankle: str


@dataclass(frozen=True)
class RobotProfile:
    name: str
    joint_order: tuple[str, ...]
    home: dict
    limits: dict
    stand_height: float
    legs: dict  # side -> LegJoints
    hip_roll_joints: tuple[str, str]
    head_bob_joint: str
    head_sway_joint: str
    # Multiplying a (hip_pitch, knee, -ankle) delta by lift_sign[side]
    # SHORTENS that leg (FK-validated).
    lift_sign: dict = field(default_factory=dict)
    # Multiplying a hip-roll delta by sway_sign[toward] leans the body over
    # that side's leg (FK-validated).
    sway_sign: dict = field(default_factory=dict)
    # Base rise during a single-leg lift so the planted foot is not pressed
    # through the floor (FK-validated compensation).
    support_rise: float = 0.0
    soft_limit_factor: float = 0.9


MICRODUCK = RobotProfile(
    name="microduck",
    joint_order=(
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ),
    home={
        "left_hip_yaw": 0.0,
        "left_hip_roll": -0.0873,
        "left_hip_pitch": -0.4579,
        "left_knee": -0.0049,
        "left_ankle": 0.4530,
        "neck_pitch": 0.3491,
        "head_pitch": 0.3491,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0873,
        "right_hip_pitch": 0.4579,
        "right_knee": 0.0049,
        "right_ankle": -0.4530,
    },
    limits={
        "left_hip_yaw": (-0.436, 0.524),
        "left_hip_roll": (-0.384, 0.384),
        "left_hip_pitch": (-1.571, 1.571),
        "left_knee": (-1.571, 1.571),
        "left_ankle": (-1.571, 1.571),
        "neck_pitch": (-1.571, 1.047),
        "head_pitch": (-1.571, 1.571),
        "head_yaw": (-2.967, 2.967),
        "head_roll": (-0.436, 0.436),
        "right_hip_yaw": (-0.524, 0.436),
        "right_hip_roll": (-0.384, 0.384),
        "right_hip_pitch": (-1.571, 1.571),
        "right_knee": (-1.571, 1.571),
        "right_ankle": (-1.571, 1.571),
    },
    stand_height=0.115,
    legs={
        "left": LegJoints("left_hip_pitch", "left_knee", "left_ankle"),
        "right": LegJoints("right_hip_pitch", "right_knee", "right_ankle"),
    },
    hip_roll_joints=("left_hip_roll", "right_hip_roll"),
    head_bob_joint="neck_pitch",
    head_sway_joint="head_yaw",
    lift_sign={"right": -1.0, "left": 1.0},
    sway_sign={"left": -1.0, "right": 1.0},
    support_rise=0.006,
)

PROFILES = {"microduck": MICRODUCK}


def get_profile(name: str) -> RobotProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"Unknown robot profile {name!r}; available: {sorted(PROFILES)}"
        ) from None
