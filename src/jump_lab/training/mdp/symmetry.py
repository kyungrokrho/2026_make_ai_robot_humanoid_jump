"""Left-right mirror of observations and actions for RSL-RL's symmetry extension.

The task is mirror symmetric about the sagittal plane, so a policy that answers a
mirrored state with the mirrored action loses nothing, and from a symmetric stance
it jumps symmetrically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .jump_command import mirror_joints

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Index and sign per observation term. "joints" is filled in from the joint names.
_SWAP = ([1, 0], [1.0, 1.0])
_TERMS: dict[str, tuple[list[int], list[float]] | str | None] = {
  "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),
  "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),
  "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
  "joint_pos": "joints",
  "joint_vel": "joints",
  "actions": "joints",
  "pelvis_height": None,
  "feet_contact": _SWAP,
  "jump_cue": None,
  "phase": None,
  "jump_time": None,
  "jump_record": None,
  "landing_error": ([0, 1, 2, 3], [1.0, -1.0, -1.0, 1.0]),
}


def build_maps(env: ManagerBasedRlEnv) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  """Mirror index and sign for every observation group and for the action."""
  joints = mirror_joints(tuple(env.scene["robot"].joint_names))
  om = env.observation_manager
  maps = {}
  for group, names in om.active_terms.items():
    index: list[int] = []
    sign: list[float] = []
    for name, shape in zip(names, om.group_obs_term_dim[group], strict=True):
      spec = _TERMS[name]  # A new observation term must be given a mirror rule.
      if spec is None:
        spec = (list(range(shape[0])), [1.0] * shape[0])
      elif spec == "joints":
        spec = joints
      assert len(spec[0]) == shape[0], (name, shape)
      index += [len(index) + i for i in spec[0]]
      sign += spec[1]
    maps[group] = (index, sign)
  maps["action"] = joints
  return {
    k: (torch.tensor(i, device=env.device), torch.tensor(s, device=env.device))
    for k, (i, s) in maps.items()
  }


def _maps(env: Any) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  base = env.unwrapped
  if not hasattr(base, "_mirror_maps"):
    base._mirror_maps = build_maps(base)
  return base._mirror_maps


def mirror(env: Any, obs: Any = None, actions: torch.Tensor | None = None):
  """Return the batch followed by its mirror image: ``[N, ...] -> [2N, ...]``."""
  maps = _maps(env)
  if obs is not None:
    mirrored = obs.clone()
    for group in obs.keys():
      index, sign = maps[group]
      mirrored[group] = obs[group][:, index] * sign
    obs = torch.cat([obs, mirrored], dim=0)
  if actions is not None:
    index, sign = maps["action"]
    actions = torch.cat([actions, actions[:, index] * sign], dim=0)
  return obs, actions
