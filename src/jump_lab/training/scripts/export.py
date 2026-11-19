"""Export a training checkpoint as a self-contained CPU deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

import jump_lab.training.tasks  # noqa: F401
from jump_lab.training.ablations import make_env
from jump_lab.training.rl_cfg import jump_ppo_runner_cfg
from jump_lab.training.runner import JumpRunner
from jump_lab.variants import VARIANTS


def export(checkpoint: Path, output: Path, variant: str, device="cuda:0"):
  if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"Use an empty output directory: {output}")
  cfg = make_env(variant, "play")
  cfg.seed = 0
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  agent = jump_ppo_runner_cfg(variant)
  raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
  try:
    runner = JumpRunner(env, asdict(agent), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=device)
    obs, _ = env.reset()
    output.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(output), "policy.onnx")
    raw.scene.write(output / "robot")
    # File-backed mesh assets are not in MjSpec.assets. Copy the G1 sources too.
    for mesh in ET.parse(output / "robot/scene.xml").getroot().findall("asset/mesh"):
      relative = Path(mesh.attrib["file"])
      destination = output / "robot/assets" / relative
      if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(G1_XML.parent / "assets" / relative, destination)
    # Scene.write serializes the spec; simulation options are applied later by MjLab.
    model = raw.sim.mj_model
    options = {}
    for name in dir(model.opt):
      if name.startswith("_"):
        continue
      value = getattr(model.opt, name)
      if isinstance(value, np.ndarray):
        options[name] = value.tolist()
      elif isinstance(value, (int, float)):
        options[name] = value
      elif hasattr(value, "__int__"):
        options[name] = int(value)
    robot = raw.scene["robot"]
    joints = [f"robot/{name}" for name in robot.joint_names]
    joint_ids = [model.joint(name).id for name in joints]
    ctrl_ids = [
      int(np.flatnonzero(model.actuator_trnid[:, 0] == j)[0]) for j in joint_ids
    ]
    action = raw.action_manager.get_term("joint_pos")
    scale = action._scale
    if isinstance(scale, torch.Tensor):
      scale = scale[0].cpu().tolist()
    meta = {
      "format_version": 1,
      "variant": variant,
      "iteration": int(checkpoint.stem.removeprefix("model_")),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "mujoco_version": mujoco.__version__,
      "observation_size": 103,
      "action_size": 29,
      "decimation": cfg.decimation,
      "physics": options,
      "joint_names": joints,
      "actuator_ids": ctrl_ids,
      "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
      "action_scale": scale,
      "clip_actions": agent.clip_actions,
      "initial_qpos": raw.sim.data.qpos[0].cpu().tolist(),
      "fsm": asdict(cfg.commands["jump"].fsm),
      "contact_force_threshold": cfg.commands["jump"].contact_force_threshold,
      "observation_terms": runner.identity()["observations"]["actor"],
    }
    (output / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Compare normalized deterministic inference on actual standing/jumping observations.
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
      str(output / "policy.onnx"),
      sess_options=options,
      providers=["CPUExecutionProvider"],
    )
    policy = runner.get_inference_policy(device=device)
    term = raw.command_manager.get_term("jump")
    max_error = 0.0
    reference = []
    with torch.inference_mode():
      for i in range(300):
        action_t = policy(obs)
        result = session.run(
          None, {session.get_inputs()[0].name: obs["actor"].cpu().numpy()}
        )[0]
        error = float(np.max(np.abs(result - action_t.cpu().numpy())))
        max_error = max(max_error, error)
        np.testing.assert_allclose(result, action_t.cpu().numpy(), atol=2e-5, rtol=2e-5)
        if i == 99:
          term.fsm.pending_target[:] = 1.0
          term.fsm.request[:] = True
        obs, _, _, _ = env.step(action_t)
        reference.append(
          {
            "time": (i + 1) * raw.step_dt,
            "phase": int(term.fsm.phase[0]),
            "pelvis_height": float(robot.data.root_link_pos_w[0, 2]),
            "landed": bool(term.fsm.landed[0]),
            "distance": float(term.fsm.delta[0]),
          }
        )
    (output / "export_validation.json").write_text(
      json.dumps(
        {
          "observations_tested": 300,
          "max_action_error": max_error,
          "reference_jump_1m": reference,
        },
        indent=2,
      )
      + "\n"
    )
    print(f"Exported {variant}: {output}; maximum ONNX action error={max_error:.3g}")
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--variant", choices=VARIANTS, default="jump")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  export(args.checkpoint, args.output, args.variant, args.device)


if __name__ == "__main__":
  main()
