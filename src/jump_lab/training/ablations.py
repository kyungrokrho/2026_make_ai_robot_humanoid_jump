"""Each ablation changes exactly one ingredient of the baseline."""

from jump_lab.variants import VARIANTS, Variant

from .env_cfg import jump_env_cfg


def make_env(variant: Variant = "jump", mode="train"):
  if variant not in VARIANTS:
    raise ValueError(f"Unknown variant: {variant}")
  cfg = jump_env_cfg(mode)
  cfg.commands["jump"].variant = variant
  if variant == "jump_no_symmetry":
    del cfg.rewards["symmetry"]  # PPO mirror loss is also disabled in rl_cfg.py.
  elif variant == "jump_no_curriculum":
    cfg.commands["jump"].fsm.delta_max = 2.0
    cfg.curriculum = {}
  elif variant == "jump_no_pose":
    del cfg.rewards["pose"]
  return cfg
