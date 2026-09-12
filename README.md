# ducktok

Write a TikTok dance as a few lines of YAML. Train a robot to perform it.

`ducktok` is a beat-grid choreography compiler for motion-imitation RL. It
turns a declarative dance spec into the reference-motion keyframes that
BeyondMimic-style tracking tasks train against, so teaching a robot a new
dance stops being custom animation code and becomes this:

```yaml
name: toosie-slide
robot: microduck
bpm: 82
bars: 2
tracks:
  - {move: lift, side: right, beats: [1]}
  - {move: sway, toward: left, beats: [1]}
  - {move: slide, direction: left, beats: [2], distance: 0.06}
  - {move: lift, side: left, beats: [3]}
  - {move: sway, toward: right, beats: [3]}
  - {move: slide, direction: right, beats: [4], distance: 0.06}
  - {move: bounce, amount: 0.004}
  - {move: head_bob, beats: [1, 2, 3, 4], amount: 0.14}
  - {move: head_sway, amount: 0.18, period_bars: 1}
```

That file is the actual Toosie Slide a
[Microduck](https://pollen-robotics.com/microduck/) learned end to end:
trained in about an hour on one L4 GPU, danced with 18.6 mm mean tracking
error and zero falls under full physics, domain randomization, and
perturbations. The complete training setup lives in
[microduck-courier](https://github.com/selinayfilizp/microduck-courier).

## Why this exists

Getting a legged robot to dance via RL is well understood in the large
(DeepMimic, BeyondMimic: track a reference motion with exp-kernel rewards).
The friction is all in the reference motion itself:

1. Authoring keyframes per dance is custom code every time.
2. Sign conventions bite on every new robot. Which direction shortens a
   leg? Which hip-roll sign leans left? Getting one wrong wastes a GPU run.
3. Nobody checks feasibility before training, so bad keyframes (a foot
   through the floor, a loop that does not close) get discovered late and
   expensively.

`ducktok` splits those concerns. Dances are declarative specs on a musical
beat grid. Robot facts live in a profile whose sign conventions are
validated ONCE with a forward-kinematics replay and never rediscovered.
Compiled clips loop seamlessly by construction, and the compiler warns when
slides do not cancel over a bar.

## Quickstart

```bash
pip install -e .
ducktok choreos/toosie_slide.yaml -o toosie_slide.csv
```

The CSV format is mjlab's `csv_to_npz` input, one row per frame at 50 fps:
`base x, y, z, quat x, y, z, w`, then joint angles in the profile's order.

## The full pipeline (Microduck reference implementation)

1. **Compile**: `ducktok choreos/your_dance.yaml -o your_dance.csv`.
2. **Validate for $0**: in the robot repo, replay the CSV kinematically
   through the real model. You get an npz for training, a rendered ghost
   video, and a per-beat feasibility report (foot lift heights, ground
   clearance, loop seam). This step caught both sign bugs in the original
   Toosie draft numerically, before any GPU spend.
3. **Train**: a BeyondMimic-style tracking task consumes the npz
   (`Mjlab-Tracking-Flat-MicroDuck` in microduck-courier, roughly an hour
   on one L4 for a 6 s loop).
4. **Film**: record the policy under full physics; the sidecar reports mean
   and p95 tracking error and fall count, with provenance (git commit,
   policy SHA-256).

Steps 2 to 4 use the scripts in
[microduck-courier/microduck_rl/scripts](https://github.com/selinayfilizp/microduck-courier/tree/main/microduck_rl):
`motion_csv_to_npz.py`, `train_courier_hf.sh` (any task id), and
`record_tracking_policy.py`.

## Move library

| move | params | what it does |
| --- | --- | --- |
| `lift` | `side`, `beats`, `amount` | pick one foot up for a beat, with support-leg rise |
| `sway` | `toward`, `beats`, `amount` | shift weight over one leg |
| `slide` | `direction`, `beats`, `distance` | travel laterally during a beat |
| `bounce` | `amount` | subtle every-beat groove bounce |
| `head_bob` | `beats`, `amount` | nod on the listed beats |
| `head_sway` | `amount`, `period_bars` | slow head yaw wave |

Amplitude defaults are FK-validated for the Microduck; every parameter can
be overridden per track. Moves compose additively and every envelope starts
and ends at rest inside its beat, so any combination loops.

## Adding a dance

Pick the BPM, break the move loop into a 4-beat bar, write the YAML, and
compile. If the compiler's loop-seam and drift checks pass, run the $0 ghost
replay and look at it. Then train. The Toosie Slide went from spec to
trained policy in one evening; the reference clip costs nothing to iterate.

## Adding a robot

Add a `RobotProfile` (`ducktok/profile.py`): joint order, standing pose,
limits, which joints form each leg, and the three sign conventions
(`lift_sign`, `sway_sign`, `support_rise`). Validate the signs with one FK
replay of a single `lift` track per side: the report tells you within
seconds if a sign is flipped (the foot goes down instead of up). Every
choreography then works on your robot unchanged, scaled by your amplitudes.

## Honest limitations

- Moves are amplitude-parameterized primitives, not arbitrary trajectories;
  a dance outside the library needs a new primitive (they are ~15 lines).
- The compiler is kinematic: dynamic feasibility is enforced by the FK
  report heuristics and ultimately by RL training, not by the compiler.
- One robot profile ships today (Microduck, 14 servos, no arms). Arm moves
  will land with the first armed profile.
- Tempo ceiling on small servo robots is real: one weight shift per beat
  around 80 to 120 BPM works; double-time footwork at 160 BPM will not.

## License

Apache-2.0.
