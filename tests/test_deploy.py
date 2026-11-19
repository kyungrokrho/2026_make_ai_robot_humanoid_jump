"""CPU-only deployment checks, including the bundled policies."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jump_lab.deploy.play import headless
from jump_lab.deploy.runtime import Playback
from jump_lab.variants import VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_import_does_not_load_training_stack():
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      "import sys; import jump_lab.deploy.play; "
      'assert not {"torch", "mjlab", "warp"}.intersection(sys.modules)',
    ],
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("variant", VARIANTS)
def test_bundled_policy_reproduces_reference_phase_sequence(variant):
  bundle = ROOT / "policies" / variant
  result = headless(Playback(bundle), 6, 2, 1)
  reference = json.loads((bundle / "export_validation.json").read_text())[
    "reference_jump_1m"
  ]
  assert [r["phase"] for r in result["frames"]] == [r["phase"] for r in reference]
  # GPU/CPU contact solvers can diverge after the no-pose policy falls over.
  height_error = max(
    abs(a["pelvis_height"] - b["pelvis_height"])
    for a, b in zip(result["frames"], reference, strict=True)
  )
  assert height_error < (0.05 if variant == "jump_no_pose" else 0.005)
  if variant == "jump":
    assert result["landed"]
    assert abs(result["landing_distance"] - 1) < 0.03


@pytest.mark.parametrize("target", [0.0, 0.5, 1.5, 2.0])
def test_baseline_lands_at_requested_distance(target):
  result = headless(Playback(ROOT / "policies/jump"), 6, 2, target)
  assert result["landed"]
  assert abs(result["landing_distance"] - target) < 0.1 + 0.05 * target


def test_request_replaces_any_phase_and_reset_is_explicit():
  p = Playback(ROOT / "policies/jump")
  for phase in range(5):
    p.state.phase = phase
    p.request(0.5)
    p.step()
    assert p.state.phase == 1
    assert p.state.time == 0
    assert p.state.target == 0.5
  before = p.steps
  p.data.qpos[2] = 0.1
  p.step()
  assert p.steps == before + 1  # No automatic reset on a fall.
  p.reset()
  assert p.steps == 0
  assert p.state.phase == 0
  np.testing.assert_allclose(p.data.qpos, p.config["initial_qpos"])


def test_reset_has_identical_initial_state_across_policies():
  states = [Playback(ROOT / "policies" / v).data.qpos.copy() for v in VARIANTS]
  for state in states[1:]:
    np.testing.assert_array_equal(states[0], state)
