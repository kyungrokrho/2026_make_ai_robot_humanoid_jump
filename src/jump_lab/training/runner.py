"""Checkpoint identity and curriculum persistence for the jump tasks and ablations."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner


class JumpRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    self.source_digest = None
    if log_dir is not None and self.gpu_global_rank == 0:
      self.source_digest = self._snapshot(Path(log_dir))

  def identity(self) -> dict:
    env = self.env.unwrapped
    om = env.observation_manager
    return {
      "variant": env.command_manager.get_term("jump").cfg.variant,
      "observations": {
        group: [
          {"name": name, "shape": list(shape)}
          for name, shape in zip(names, om.group_obs_term_dim[group], strict=True)
        ]
        for group, names in om.active_terms.items()
      },
      "default_joint_pos": env.scene["robot"].data.default_joint_pos[0].tolist(),
    }

  def _snapshot(self, log_dir: Path) -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted(
      [*root.glob("src/jump_lab/**/*.py"), *root.glob("tests/*.py")]
      + [
        root / p for p in ("pyproject.toml", "uv.lock", "README.md", "docs/training.md")
      ]
    )
    paths = [p for p in paths if p.is_file()]
    digest = hashlib.sha256()
    out = log_dir / "params"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
      for path in paths:
        relative = path.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(path.read_bytes())
        archive.add(path, arcname=str(relative))
    identity = {**self.identity(), "source_sha256": digest.hexdigest()}
    (out / "jump.json").write_text(json.dumps(identity, indent=2) + "\n")
    return digest.hexdigest()

  def save(self, path: str, infos=None) -> None:
    term = self.env.unwrapped.command_manager.get_term("jump")
    metadata = {
      **self.identity(),
      "source_sha256": self.source_digest,
      "delta_max": term.fsm.delta_max,
      "band_count": term.band_count,
      "band_success": term.band_success,
    }
    super().save(path, {**(infos or {}), "jump_lab": metadata})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    # Validate before loading any weights, including actor-only play loads. All
    # actors have 103 inputs, so tensor shape alone cannot identify the task.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = (checkpoint.get("infos") or {}).get("jump_lab")
    expected = self.identity()
    if metadata is None:
      raise ValueError("Checkpoint is missing Jump Lab metadata.")
    for key, value in expected.items():
      if metadata.get(key) != value:
        raise ValueError(f"Checkpoint {key} does not match the selected jump task.")
    del checkpoint
    infos = super().load(path, load_cfg, strict, map_location)
    # Actor-only evaluation/play must keep their fixed targets and fresh episodes.
    if load_cfg is None:
      term = self.env.unwrapped.command_manager.get_term("jump")
      term.fsm.delta_max = float(metadata["delta_max"])
      self.env.reset()
      term.band_count = int(metadata["band_count"])
      term.band_success = int(metadata["band_success"])
    return infos
