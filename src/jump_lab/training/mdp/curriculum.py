from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .jump_command import JumpCommand


def target_distance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  step: float = 0.10,
  limit: float = 2.0,
  min_episodes: int = 4096,
  promote_rate: float = 0.7,
) -> dict[str, float]:
  """Raise the upper target bound when the top band succeeds often enough.

  Success is counted on the stochastic training policy over the last
  ``min_episodes`` commanded episodes whose target was in the top band.
  """
  del env_ids
  term: JumpCommand = env.command_manager.get_term(command_name)
  rate = term.band_success / max(term.band_count, 1)
  if term.band_count >= min_episodes:
    if rate >= promote_rate and term.fsm.delta_max + step <= limit + 1e-6:
      term.fsm.delta_max += step
    term.band_count = 0
    term.band_success = 0
  return {"delta_max": term.fsm.delta_max, "band_success_rate": rate}
