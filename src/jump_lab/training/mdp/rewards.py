"""Educational jump rewards: body pose, push-off, landing and smooth motion.

Continuous rewards are per second. Event/increment rewards divide by step_dt
because MjLab's reward manager multiplies every term by step_dt.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .fsm import PHASE_FLIGHT, PHASE_PREP, JumpFsm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _fsm(env: ManagerBasedRlEnv, command_name: str) -> JumpFsm:
  term = env.command_manager.get_term(command_name)
  term.ensure_updated()
  return term.fsm


def _resting(f: JumpFsm) -> torch.Tensor:
  """Standby, recovery and done: the phases where the robot should stand still."""
  return (f.phase != PHASE_PREP) & (f.phase != PHASE_FLIGHT)


def match(error: torch.Tensor) -> torch.Tensor:
  """Nonnegative error: zero earns 1, error=1 earns 0.5."""
  return 1.0 / (1.0 + error)


def pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Track the default pose during standby, recovery and done."""
  f = _fsm(env, command_name)
  assert f.inp is not None
  return match(f.inp.pose_err / 0.30) * _resting(f)


def still(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Stand upright without translating, rotating or drifting from the anchor."""
  f = _fsm(env, command_name)
  assert f.inp is not None
  inp = f.inp
  drift = (inp.pelvis_pos[:, :2] - f.anchor_xy).norm(dim=1)
  cost = (
    (inp.torso_tilt / math.radians(12)).square().clamp(max=1)
    + (inp.lin_vel.norm(dim=1) / 0.15).square().clamp(max=1)
    + (inp.ang_vel.norm(dim=1) / 0.5).square().clamp(max=1)
    + (drift / 0.10).square().clamp(max=1)
  )
  return cost * _resting(f)


def alive(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """make surviving the standing phases worthwhile during exploration."""
  return _resting(_fsm(env, command_name)).float()


def prep_time(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """allow time to crouch without losing the standing reward immediately."""
  f = _fsm(env, command_name)
  return ((f.phase == PHASE_PREP) & (f.t_cmd < 0.6)).float()


def crouch(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """pay only for new depth records, balanced on both feet, up to 0.15 m."""
  f = _fsm(env, command_name)
  assert f.drop_gain is not None
  return f.drop_gain / f.cfg.crouch_depth / env.step_dt


def push(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Pay only for new upward-speed records, up to 2.5 m/s."""
  f = _fsm(env, command_name)
  return f.vz_gain / f.vz_target / env.step_dt


def land(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Once at landing: accurate distance, straight direction and enough height."""
  f = _fsm(env, command_name)
  distance = match((f.err_x / 0.15).square())
  direction = match((f.err_y / 0.10).square() + (f.err_yaw / math.radians(20)).square())
  height = (f.apex / f.apex_target).clamp(0, 1)
  return f.landing_event * distance * direction * (0.4 + 0.6 * height) / env.step_dt


def symmetry(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Penalize left-right differences during preparation and flight."""
  f = _fsm(env, command_name)
  assert f.inp is not None
  jumping = (f.phase == PHASE_PREP) | (f.phase == PHASE_FLIGHT)
  return (1.0 - match(f.inp.asymmetry / 0.40)) * jumping


def smooth(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Use the same action-change penalty for every joint and phase."""
  del command_name
  am = env.action_manager
  return (am.action - am.prev_action).square().sum(dim=1)


def fail(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Pay once for any failure termination, excluding the six-second time limit."""
  del command_name
  return env.termination_manager.terminated.float() / env.step_dt
