"""Single-robot CPU playback. No MjLab, CUDA, Torch or training code imports.

The exported XML contains native MuJoCo position actuators (including PD gains,
armature and force limits). We supply position targets, not explicit torques.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

PHASE_NAMES = ("Standby", "Prepare", "Flight", "Recovery", "Done")
HEEL = np.array([-0.064, 0.0, -0.035])
TOE = np.array([0.142, 0.0, -0.035])


class JumpState:
  """Only the phase state needed by the actor and floor markers.

  Reward records and training termination checks do not affect the actor's cue.
  Playback never terminates or resets automatically. Requests replace any phase.
  """

  def __init__(self, config):
    self.cfg = config
    self.phase = 0
    self.target = 1.0
    self.time = 0.0
    self.air_ticks = 0
    self.stable_time = 0.0
    self.pending = None
    self.commanded = False
    self.took_off = False
    self.landed = False
    self.landing_distance = None
    self.origin = np.zeros(2)
    self.rotation = np.eye(2)
    self.start = np.zeros(2)
    self.com0 = np.zeros(3)

  def frame(self, xy):
    return (xy - self.origin) @ self.rotation

  def anchor(self, inputs):
    self.origin = inputs["pelvis"][:2].copy()
    yaw = inputs["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    self.rotation = np.array([[c, -s], [s, c]])
    self.start = np.array(
      [
        self.frame(inputs["heel"][:, :2])[:, 0].min(),
        self.frame(inputs["foot"][:, :2])[:, 1].mean(),
      ]
    )

  def update(self, inp, contact, dt):
    cfg = self.cfg
    old = self.phase
    both = bool(contact.all())
    stable = (
      both
      and np.linalg.norm(inp["linear_velocity"]) < cfg["stable_lin_vel"]
      and np.linalg.norm(inp["angular_velocity"]) < cfg["stable_ang_vel"]
      and inp["tilt"] < cfg["stable_tilt"]
      and inp["torso_tilt"] < cfg["stable_tilt"]
    )
    self.stable_time = self.stable_time + dt if stable else 0.0
    self.air_ticks = self.air_ticks + 1 if not contact.any() else 0
    if self.pending is not None:
      self.target = self.pending
      self.pending = None
      self.phase = 1
      self.time = 0.0
      self.air_ticks = 0
      self.stable_time = 0.0
      self.took_off = self.landed = False
      self.landing_distance = None
      self.commanded = True
      self.com0 = inp["com"].copy()
      self.anchor(inp)
      return
    if old >= 1:
      self.time += dt
    clearance = min(inp["heel"][:, 2].min(), inp["toe"][:, 2].min())
    airborne = (
      self.air_ticks >= cfg["takeoff_air_ticks"]
      and clearance >= cfg["takeoff_clearance"]
    )
    if (
      old == 1
      and airborne
      and inp["com"][2] - self.com0[2] >= cfg["takeoff_min_rise"]
      and inp["com_velocity"][2] > 0
      and inp["tilt"] < cfg["takeoff_max_tilt"]
    ):
      self.phase = 2
      self.took_off = True
    elif old == 2 and both:
      self.phase = 3
      self.stable_time = 0.0
      self.air_ticks = 0
      self.landed = True
      self.landing_distance = float(
        self.frame(inp["heel"][:, :2])[:, 0].min() - self.start[0]
      )
    elif old == 3 and not airborne and self.stable_time >= cfg["recovery_stable_s"]:
      self.phase = 4
    elif old == 4 and stable:
      self.phase = 0
      self.stable_time = 0.0

  def cue(self):
    return (
      np.array(
        [self.phase == 1, self.phase == 2, self.target, self.time], dtype=np.float32
      )
      if self.phase in (1, 2)
      else np.zeros(4, dtype=np.float32)
    )


class Playback:
  def __init__(self, bundle: Path):
    self.bundle = Path(bundle).resolve()
    self.config = json.loads((self.bundle / "config.json").read_text())
    if self.config["format_version"] != 1:
      raise ValueError("Unsupported policy bundle version")
    self.model = mujoco.MjModel.from_xml_path(str(self.bundle / "robot/scene.xml"))
    for name, value in self.config["physics"].items():
      setattr(self.model.opt, name, value)
    self.data = mujoco.MjData(self.model)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    self.policy = ort.InferenceSession(
      str(self.bundle / "policy.onnx"),
      sess_options=opts,
      providers=["CPUExecutionProvider"],
    )
    self.input_name = self.policy.get_inputs()[0].name
    if self.policy.get_inputs()[0].shape != [1, 103] or self.policy.get_outputs()[
      0
    ].shape != [1, 29]:
      raise ValueError("Expected a 103-input, 29-output policy")
    m = self.model
    self.joint_ids = np.array([m.joint(name).id for name in self.config["joint_names"]])
    self.qpos_ids = m.jnt_qposadr[self.joint_ids]
    self.qvel_ids = m.jnt_dofadr[self.joint_ids]
    self.ctrl_ids = np.array(self.config["actuator_ids"])
    self.default_pose = np.array(self.config["default_joint_pos"])
    self.scale = np.array(self.config["action_scale"])
    self.pelvis_id = m.body("robot/pelvis").id
    self.torso_id = m.body("robot/torso_link").id
    self.foot_ids = np.array(
      [m.body(f"robot/{s}_ankle_roll_link").id for s in ("left", "right")]
    )
    self.robot_ids = np.array(
      [i for i in range(m.nbody) if m.body(i).name.startswith("robot/")]
    )
    self.masses = m.body_mass[self.robot_ids]
    self.decimation = self.config["decimation"]
    self.dt = m.opt.timestep * self.decimation
    self.threshold = self.config["contact_force_threshold"]
    self.reset()

  def reset(self):
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] = self.config["initial_qpos"]
    self.action = np.zeros(29, dtype=np.float32)
    self.data.ctrl[self.ctrl_ids] = self.default_pose
    mujoco.mj_forward(self.model, self.data)
    self.state = JumpState(self.config["fsm"])
    self.state.anchor(self.inputs())
    self.contact = self.foot_forces() > self.threshold
    self.steps = 0

  def request(self, target):
    if not np.isfinite(target) or not 0 <= target <= 2:
      raise ValueError("Target distance must be between 0 and 2 metres")
    self.state.pending = float(target)

  def foot_forces(self):
    """Net ground force per foot, excluding robot self contacts."""
    return np.array(
      [
        np.linalg.norm(
          self.data.sensor(f"feet_ground_contact_{side}_ankle_roll_link_force").data
        )
        for side in ("left", "right")
      ]
    )

  def inputs(self):
    d = self.data
    pelvis = d.xpos[self.pelvis_id]
    rotation = d.xmat[self.pelvis_id].reshape(3, 3)
    foot = d.xpos[self.foot_ids]
    rotations = d.xmat[self.foot_ids].reshape(2, 3, 3)
    centre = d.subtree_com[self.pelvis_id]
    velocities = d.cvel[self.robot_ids, 3:] + np.cross(
      d.cvel[self.robot_ids, :3], d.xipos[self.robot_ids] - centre
    )
    root_vel = d.cvel[self.pelvis_id, 3:] + np.cross(
      d.cvel[self.pelvis_id, :3], pelvis - centre
    )
    return {
      "pelvis": pelvis.copy(),
      "foot": foot.copy(),
      "heel": foot + rotations @ HEEL,
      "toe": foot + rotations @ TOE,
      "yaw": np.arctan2(rotation[1, 0], rotation[0, 0]),
      "tilt": np.arccos(np.clip(rotation[2, 2], -1, 1)),
      "torso_tilt": np.arccos(np.clip(d.xmat[self.torso_id, 8], -1, 1)),
      "linear_velocity": root_vel,
      "angular_velocity": d.cvel[self.pelvis_id, :3].copy(),
      "com": np.average(d.xipos[self.robot_ids], axis=0, weights=self.masses),
      "com_velocity": np.average(velocities, axis=0, weights=self.masses),
    }

  def observation(self):
    d = self.data
    # Use the exported pelvis IMU site, not root qvel (different location/frame).
    obs = np.concatenate(
      [
        d.sensor("robot/imu_lin_vel").data,
        d.sensor("robot/imu_ang_vel").data,
        -d.xmat[self.pelvis_id].reshape(3, 3)[2, :],
        d.qpos[self.qpos_ids] - self.default_pose,
        d.qvel[self.qvel_ids],
        self.action,
        [d.xpos[self.pelvis_id, 2]],
        self.contact,
        self.state.cue(),
      ]
    ).astype(np.float32)
    if obs.shape != (103,) or not np.isfinite(obs).all():
      raise RuntimeError("Invalid policy observation; reset the simulation")
    return obs

  def step(self):
    action = self.policy.run(None, {self.input_name: self.observation()[None]})[0][0]
    if not np.isfinite(action).all():
      raise RuntimeError("Policy produced non-finite actions")
    clip = self.config["clip_actions"]
    self.action = action if clip is None else np.clip(action, -clip, clip)
    self.data.ctrl[self.ctrl_ids] = self.default_pose + self.scale * self.action
    history = []
    for _ in range(self.decimation):
      mujoco.mj_step(self.model, self.data)
      history.append(self.foot_forces())
    # Match MjLab: FSM sees the final substep's pre-integration derived state.
    self.state.update(self.inputs(), np.max(history, axis=0) > self.threshold, self.dt)
    mujoco.mj_forward(self.model, self.data)
    # MjLab's contact sensor cache contains the last substep sample, while IMU
    # readings and kinematics have been refreshed by forward().
    self.contact = history[-1] > self.threshold
    self.steps += 1

  def markers(self, preview_target=None):
    state = self.state
    inp = self.inputs()
    local = np.array(
      [
        state.frame(inp["heel"][:, :2])[:, 0].min(),
        state.frame(inp["foot"][:, :2])[:, 1].mean(),
      ]
    )
    target = (
      state.target if state.commanded or preview_target is None else preview_target
    )
    goal = state.origin + state.rotation @ (state.start + [target, 0])
    current = state.origin + state.rotation @ local
    return goal, current, float(local[0] - state.start[0])
