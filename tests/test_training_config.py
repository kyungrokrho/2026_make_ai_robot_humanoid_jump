"""Optional training tests: run with the train extra installed."""

from dataclasses import asdict

import pytest

pytest.importorskip("mjlab")

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import jump_lab.training.tasks  # noqa: F401
from jump_lab.variants import VARIANTS, task_id


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["train", "eval", "play"])
def test_only_intended_ablation_differences(variant, mode):
  kwargs = {"play": mode == "play"}
  base = load_env_cfg(task_id("jump", evaluation=mode == "eval"), **kwargs)
  other = load_env_cfg(task_id(variant, evaluation=mode == "eval"), **kwargs)
  expected = asdict(base)
  expected["commands"]["jump"]["variant"] = variant
  if variant == "jump_no_symmetry":
    del expected["rewards"]["symmetry"]
  if variant == "jump_no_pose":
    del expected["rewards"]["pose"]
  if variant == "jump_no_curriculum":
    expected["commands"]["jump"]["fsm"]["delta_max"] = 2.0
    expected["curriculum"] = {}
  actual = asdict(other)
  for cfg in (expected, actual):
    cfg["scene"]["terrain"]["spec_fn"] = cfg["scene"]["terrain"]["spec_fn"]().to_xml()
  assert actual == expected
  if mode == "play":
    assert not other.terminations
  else:
    assert len(other.terminations) == 10


@pytest.mark.parametrize("variant", VARIANTS)
def test_ppo_only_changes_symmetry_and_experiment_name(variant):
  expected = asdict(load_rl_cfg(task_id("jump")))
  actual = asdict(load_rl_cfg(task_id(variant)))
  expected["experiment_name"] = variant
  if variant == "jump_no_symmetry":
    expected["algorithm"]["symmetry_cfg"] = None
  assert actual == expected
