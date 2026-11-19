"""Unitree G1 standing long jump environment."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from jump_lab.training import mdp
from jump_lab.training.mdp.fsm import FAIL_NAMES, JumpFsmCfg
from jump_lab.training.mdp.rewards import (
  alive,
  crouch,
  fail,
  land,
  pose,
  prep_time,
  push,
  smooth,
  still,
  symmetry,
)

CMD = "jump"
# Knee angle of the standing pose in rad. MjLab's own pose has 0.669; the deeper
# stance reached the longest jump in the pilots.
STANCE_KNEE = 1.0


def robot_cfg():
  """G1 with the standing pose bent to ``STANCE_KNEE``, feet flat, hips over ankles."""
  cfg = get_g1_robot_cfg()
  if STANCE_KNEE != 0.669:
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos[".*_hip_pitch_joint"] = -STANCE_KNEE / 2
    joint_pos[".*_knee_joint"] = STANCE_KNEE
    joint_pos[".*_ankle_pitch_joint"] = -STANCE_KNEE / 2
    # Thigh and shank are 0.3 m each.
    drop = 0.3 * (math.cos(0.312) + math.cos(0.357) - 2 * math.cos(STANCE_KNEE / 2))
    x, y, z = cfg.init_state.pos
    cfg.init_state = replace(cfg.init_state, pos=(x, y, z - drop), joint_pos=joint_pos)
  return cfg


FEET = r"^(left_ankle_roll_link|right_ankle_roll_link)$"


def jump_env_cfg(
  mode: Literal["train", "eval", "play"] = "train",
) -> ManagerBasedRlEnvCfg:
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=FEET, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  body_ground = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "pelvis_height": ObservationTermCfg(func=mdp.pelvis_height),
    "feet_contact": ObservationTermCfg(
      func=mdp.feet_contact, params={"command_name": CMD}
    ),
    "jump_cue": ObservationTermCfg(func=mdp.jump_cue, params={"command_name": CMD}),
  }
  critic_terms = {
    **actor_terms,
    "phase": ObservationTermCfg(func=mdp.jump_phase, params={"command_name": CMD}),
    "jump_time": ObservationTermCfg(func=mdp.jump_time, params={"command_name": CMD}),
    "landing_error": ObservationTermCfg(
      func=mdp.landing_error, params={"command_name": CMD}
    ),
    "jump_record": ObservationTermCfg(
      func=mdp.jump_record, params={"command_name": CMD}
    ),
  }

  def event(weight: float, func, **params) -> RewardTermCfg:
    return RewardTermCfg(
      func=func, weight=weight, params={"command_name": CMD, **params}
    )

  rewards = {
    "pose": event(3.0, pose),
    "still": event(-0.25, still),
    "alive": event(1.0, alive),
    "prep_time": event(4.0, prep_time),
    "crouch": event(2.0, crouch),
    "push": event(1.0, push),
    "land": event(5.0, land),
    "symmetry": event(-2.0, symmetry),
    "smooth": event(-0.1, smooth),
    "fail": event(-2.0, fail),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(65.0)}
    ),
    "pelvis_low": TerminationTermCfg(
      func=mdp.root_height_below_minimum, params={"minimum_height": 0.25}
    ),
    "body_ground_contact": TerminationTermCfg(
      func=mdp.ground_contact, params={"sensor_name": body_ground.name}
    ),
  }
  for code, name in FAIL_NAMES.items():
    terminations[name] = TerminationTermCfg(
      func=mdp.jump_failure, params={"command_name": CMD, "code": code}
    )

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  curriculum = {
    "target_distance": CurriculumTermCfg(
      func=mdp.target_distance, params={"command_name": CMD}
    )
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg()},
      sensors=(feet_ground, body_ground),
      num_envs=8192,
      env_spacing=3.0,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True),
      "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands={
      CMD: mdp.JumpCommandCfg(fsm=JumpFsmCfg(stable_scale=2.0, stable_filter_s=0.2))
    },
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50
      ),
    ),
    decimation=4,
    episode_length_s=6.0,
  )

  jump = cfg.commands[CMD]
  assert isinstance(jump, mdp.JumpCommandCfg)
  if mode == "eval":
    # Mean actions, original stillness thresholds, one commanded jump per episode.
    jump.fsm.stable_scale = 1.0
    jump.fsm.stable_filter_s = 0.0
    jump.fsm.idle_fraction = 0.0
    jump.record_episodes = True
    cfg.curriculum = {}
  elif mode == "play":
    # Button driven: stay in the episode after a jump and wait for the next request.
    jump.fsm.stable_scale = 1.0
    jump.fsm.stable_filter_s = 0.0
    jump.fsm.mode = "button"
    cfg.curriculum = {}
    # Playback never auto-resets for task failures or elapsed time.
    cfg.terminations = {}
    cfg.episode_length_s = int(1e9)
    cfg.scene.num_envs = 1
    cfg.events["reset_base"].params["pose_range"] = {}
  return cfg
