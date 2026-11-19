from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

from .fsm import JumpFsm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _fsm(env: ManagerBasedRlEnv, command_name: str) -> JumpFsm:
  term = env.command_manager.get_term(command_name)
  term.ensure_updated()
  return term.fsm


def jump_failure(env: ManagerBasedRlEnv, command_name: str, code: int) -> torch.Tensor:
  return _fsm(env, command_name).fail_code == code


def ground_contact(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
  """Any body other than the feet touching the ground."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force = data.force_history.norm(dim=-1).amax(dim=(1, 2))
  else:
    assert data.force is not None
    force = data.force.norm(dim=-1).amax(dim=1)
  return force > force_threshold
