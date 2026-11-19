from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.entity import Entity

from .fsm import NUM_PHASES, PHASE_FLIGHT, PHASE_PREP, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .jump_command import JumpCommand


def _term(env: ManagerBasedRlEnv, command_name: str) -> JumpCommand:
  return env.command_manager.get_term(command_name)


def jump_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return F.one_hot(_term(env, command_name).fsm.phase, NUM_PHASES).float()


def jump_cue(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """What the actor needs to know about the jump: ``[preparing, flying, target,
  time since the command]``, all zero from landing on.

  Standby, recovery and done ask for the same thing, standing still in the default
  pose, so the actor is not told them apart. Told apart, its input jumps when the
  recovery is declared complete and the robot twitches.
  """
  f = _term(env, command_name).fsm
  prep = f.phase == PHASE_PREP
  jumping = (prep | (f.phase == PHASE_FLIGHT)).float()
  return torch.stack(
    [prep.float(), jumping - prep.float(), f.target * jumping, f.t_cmd * jumping],
    dim=1,
  )


def pelvis_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 2:3]


def feet_contact(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = _term(env, command_name)
  force = term.sensor.data.force
  assert force is not None
  return (force.norm(dim=-1) > term.cfg.contact_force_threshold).float()


def jump_time(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The critic also sees elapsed command time during recovery and done."""
  f = _term(env, command_name).fsm
  return (f.t_cmd * f.commanded)[:, None]


def landing_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current landing errors; freeze them at first landing, zero before a command."""
  term = _term(env, command_name)
  f = term.fsm
  inp = term.gather_inputs()
  heel_x = f.to_frame(inp.heel_pos[..., :2])[..., 0].min(dim=1).values
  foot_y = f.to_frame(inp.foot_pos[..., :2])[..., 1].mean(dim=1)
  ex = torch.where(f.landed, f.err_x, heel_x - f.heel0_x - f.target)
  ey = torch.where(f.landed, f.err_y, foot_y - f.feet0_y)
  yaw = torch.where(f.landed, f.err_yaw, wrap_to_pi(inp.yaw - f.yaw0))
  return torch.stack([ex, ey, yaw.sin(), yaw.cos()], dim=1) * f.commanded[:, None]


def jump_record(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Episode records used by the rewards, exposed only to the critic."""
  f = _term(env, command_name).fsm
  return torch.stack([f.drop_peak, f.vz_peak, f.apex], dim=1)
