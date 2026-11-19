"""MjLab registration; imported only by training tools."""

from mjlab.tasks.registry import register_mjlab_task

from jump_lab.training.ablations import make_env
from jump_lab.training.rl_cfg import jump_ppo_runner_cfg
from jump_lab.training.runner import JumpRunner
from jump_lab.variants import VARIANTS, task_id

for variant in VARIANTS:
  for evaluation in (False, True):
    register_mjlab_task(
      task_id=task_id(variant, evaluation=evaluation),
      env_cfg=make_env(variant, "eval" if evaluation else "train"),
      play_env_cfg=make_env(variant, "eval" if evaluation else "play"),
      rl_cfg=jump_ppo_runner_cfg(variant),
      runner_cls=JumpRunner,
    )
