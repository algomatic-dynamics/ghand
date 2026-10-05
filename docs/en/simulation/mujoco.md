# MuJoCo

The MuJoCo backend is available to teleoperation, dataset capture, and policy
rollout.

## Install

```bash
uv sync --extra sim
```

## Teleoperate in simulation

```bash
uv run --extra sim ghand teleop \
  --mujoco \
  --mujoco-viewer \
  --mujoco-scene turn_handle
```

`--mujoco-scene` accepts a built-in label (`cube_face_up` or `turn_handle`) or a
custom scene XML path.

## Generated model cache

The packaged right-hand URDF is converted to MJCF on first use and cached under:

```text
~/.cache/ghand/mujoco/<version>/
```

The built-in scene cache fingerprint includes the complete
`MujocoPhysicsProfile` and every versioned collision-asset hash. A physics-profile
or collision-asset change therefore produces a separate cached model rather than
silently reusing an incompatible scene. The cache manifest records the resolved
profile, collision asset version, source hashes, and generated assets.

## Collision geometry

MuJoCo collides with the convex hull of each mesh geom. The exporter therefore
does not combine a complete fixed-link assembly into one large hull. It preserves
each original URDF visual-mesh placement, maps it to one or more offline-generated
convex parts, and attaches all of those parts to the assembly's logical moving
body.

The packaged `v1` collision asset set under `ghand/assets/collision/mujoco/v1/`
contains 43 shared convex meshes for the 25 PLY files referenced by the packaged
URDF. The loader validates it against the current URDF, the source PLY hashes, and
the per-part hashes before every export; no download or runtime generation is
involved. Five source meshes with the largest false convex volume use a bounded
CoACD decomposition; the other source meshes use one convex hull each. Repeated
URDF placements reuse those assets, producing 171 collision geom instances. The
103 public self-collision link pairs expand to 9,512 physical geom pairs inside
MJCF. This includes every convex-part combination for the middle-ring link pair
observed during the real-hardware pre-grasp approach. The task reports contact
count and maximum penetration as diagnostics, without using them in the reward.

To regenerate the set after intentionally changing the packaged URDF or source
meshes, run:

```bash
uv run --group collision-assets python scripts/generate_mujoco_collision_assets.py
```

The command replaces the packaged version directory under
`src/ghand/assets/collision/mujoco/` in place; commit it together with the URDF or
mesh change. Generated assets record the generator
versions and their actual file hashes, which also enter the model cache and
training metadata. A custom `--urdf path/to/hand.urdf` must provide the matching
`path/to/hand.mujoco-collision.json` sidecar; its part paths are resolved relative
to that manifest.

## Control profiles

MJCF export and runtime control use the same versioned profile type and
packaged joint mapping. Interactive MuJoCo teleoperation and LeRobot
capture/rollout use `TELEOP_MUJOCO_PHYSICS_PROFILE`.

The teleop profile uses `kp=20`, a `1.0 N m` actuator effort limit, joint damping
`0.06`, and armature `0.0001`. These values were selected from collision-isolated
canonical-pose step and continuous-transition checks. They define a responsive
simulation control profile; they are not measured on the real hand, so do not
treat MuJoCo results as evidence about the physical hardware.

Each profile defines the physics timestep, control rate, target slew limit,
actuator stiffness and effort, collision geometry, and contact parameters. Joint
position limits are loaded from the packaged URDF instead of being duplicated in
the profile.

Actions pass through joint limits, velocity limiting, and Dynamixel position
quantization before the simulator is stepped. Normalized teleop targets keep the
same canonical angle mapping as the fake and real backends; source-URDF ranges are
applied afterward as safety clamps rather than rescaling the target. The exact
current profile fields and defaults are generated in the
[simulation API](../reference/python-api/simulation.md).

The default profile enables 103 validated cross-link hand self-collision pairs.
They receive the harder `solref`/`solimp` response and normally use a `1.5 mm`
proximity margin; cube, floor, and turn-handle contacts keep their existing
settings. All enabled middle-ring pairs, plus one additional task-clearance
pair, use zero margin, so their hard response begins at actual contact without
an artificial clearance force.

## Export MJCF manually

Use an explicit export when inspecting or customizing the model outside the
versioned runtime cache:

```bash
uv run --extra sim ghand export mujoco \
  --out models/ghand_mujoco_export \
  --validate
```

The output contains the hand MJCF and the built-in scene XML files. `--validate`
loads the result with MuJoCo before the command returns.

## Related workflows

- [Record a MuJoCo LeRobot dataset](../data-and-learning/lerobot-dataset.md#record-from-mujoco)
- [Train a LeRobot policy](../data-and-learning/training.md)
- [Roll out a policy](../data-and-learning/policy-rollout.md)
