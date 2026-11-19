"""The baseline and three independent training ablations."""

from typing import Literal

Variant = Literal["jump", "jump_no_symmetry", "jump_no_curriculum", "jump_no_pose"]
DEFAULT_VARIANT: Variant = "jump"
VARIANTS = ("jump", "jump_no_symmetry", "jump_no_curriculum", "jump_no_pose")
TASK_IDS = {v: "JumpLab-" + v for v in VARIANTS}


def task_id(variant: Variant, *, evaluation: bool = False) -> str:
  return TASK_IDS[variant] + ("-Eval" if evaluation else "")
