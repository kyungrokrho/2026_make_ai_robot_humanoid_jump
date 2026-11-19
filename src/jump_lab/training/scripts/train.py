"""Train the jump policy. Thin wrapper over MjLab's trainer that fixes the task and
names the run.

  uv run jump-train
  uv run jump-train --agent.run-name my_tag --env.scene.num-envs 16384
  uv run jump-train --agent.logger tensorboard --agent.max-iterations 50
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401
import tyro
from mjlab.scripts.train import TrainConfig, launch_training

import jump_lab.training.tasks  # noqa: F401
from jump_lab.variants import DEFAULT_VARIANT, Variant, task_id


@dataclass
class VariantArgs:
  variant: Variant = DEFAULT_VARIANT
  """Baseline or one of three independent ablations."""


@dataclass(frozen=True)
class JumpTrainConfig(TrainConfig):
  variant: Variant = DEFAULT_VARIANT


def _git_sha() -> str:
  try:
    out = subprocess.run(
      ["git", "rev-parse", "--short", "HEAD"],
      cwd=Path(__file__).parent,
      capture_output=True,
      text=True,
      check=True,
    )
    dirty = subprocess.run(
      ["git", "status", "--porcelain"],
      cwd=Path(__file__).parent,
      capture_output=True,
      text=True,
      check=True,
    )
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
  except (OSError, subprocess.CalledProcessError):
    return "nogit"


def main() -> None:
  selected, remaining = tyro.cli(
    VariantArgs, add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS
  )
  chosen = task_id(selected.variant)
  default = TrainConfig.from_task(chosen)
  args = tyro.cli(
    JumpTrainConfig,
    args=remaining,
    default=JumpTrainConfig(
      env=default.env, agent=default.agent, variant=selected.variant
    ),
    config=mjlab.TYRO_FLAGS,
  )
  jump = args.env.commands["jump"]
  # MjLab prefixes the timestamp, so the W&B name reads
  # 2026-11-11_12-00-00_dmax0.10_n8192_s42_<sha>[_<tag>].
  auto = (
    f"{selected.variant}_dmax{jump.fsm.delta_max:.2f}_n{args.env.scene.num_envs}"
    f"_s{args.agent.seed}_{_git_sha()}"
  )
  tag = args.agent.run_name
  args.agent.run_name = f"{auto}_{tag}" if tag else auto
  if jump.variant != selected.variant:
    raise ValueError(
      "Use --variant to select the task, not --env.commands.jump.variant."
    )
  launch_training(task_id=chosen, args=args)


if __name__ == "__main__":
  main()
