"""Evaluate the baseline and ablations with identical first-episode trials."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import jump_lab.training.tasks  # noqa: F401
from jump_lab.training.mdp import FAIL_NAMES
from jump_lab.training.runner import JumpRunner
from jump_lab.variants import DEFAULT_VARIANT, Variant, task_id


@dataclass
class Args:
  checkpoint_file: str
  variant: Variant = DEFAULT_VARIANT
  targets: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
  trials: int = 128
  """First-episode attempts per target and seed, including pre-command falls."""
  seed: int = 0
  seeds: tuple[int, ...] | None = None
  """Optional multiple evaluation seeds; overrides --seed."""
  idle: bool = False
  """Evaluate only six seconds of standing without a jump command."""
  include_idle: bool = False
  """Also evaluate standing without commands after the jump trials."""
  device: str = "cuda:0"
  output: str | None = None


def first_episodes(records, seen: torch.Tensor) -> list[dict[str, torch.Tensor]]:
  """Keep each environment's first episode, regardless of whether it got a cue."""
  rows = []
  for record in records:
    keep = ~seen[record["env_id"]]
    seen[record["env_id"]] = True
    if keep.any():
      rows.append({k: v[keep] for k, v in record.items()})
  return rows


def summarize(
  rows: list[dict[str, torch.Tensor]],
  targets: tuple[float, ...],
  trials: int,
  *,
  idle: bool = False,
) -> list[dict]:
  """Missing attempts remain in the denominator; conditional means carry counts."""
  data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else {}
  results = []
  for i, target in enumerate(targets):
    m = (
      (data["env_id"] % len(targets)) == i if data else torch.zeros(0, dtype=torch.bool)
    )
    n = int(m.sum())

    def count(key, mask=m):
      return int(data[key][mask].sum()) if data else 0

    def mean(key, condition=None, *, degrees=False, selected=m):
      if not data:
        return None
      mask = selected & torch.isfinite(data[key])
      if condition is not None:
        mask &= data[condition]
      if not mask.any():
        return None
      value = data[key][mask].mean().item()
      return math.degrees(value) if degrees else value

    fails = {
      name: int(((data["fail_code"] == code) & m).sum()) if data else 0
      for code, name in FAIL_NAMES.items()
    }
    fails["fall_or_other"] = (
      int((data["terminated"] & (data["fail_code"] == 0) & m).sum()) if data else 0
    )
    fails["unfinished"] = trials - n
    landed = m & data["landed"] if data else m
    row = {
      "target": None if idle else target,
      "attempts": trials,
      "finished": n,
      "landed_samples": count("landed"),
      "recovered_samples": count("completed"),
      "end_posture_samples": int((m & torch.isfinite(data["end_torso_tilt"])).sum())
      if data
      else 0,
      "success_rate": count("success") / trials,
      "success_survived_rate": count("success_survived") / trials,
      "success_survived_no_prep_step_rate": count("success_survived_no_prep_step")
      / trials,
      "prep_step_violation_rate": count("prep_step_violation") / trials,
      "mean_prep_contact_shift": mean("prep_contact_shift", "commanded"),
      "takeoff_rate": count("took_off") / trials,
      "landing_rate": count("landed") / trials,
      "recovery_rate": count("completed") / trials,
      "survival_rate": count("survived") / trials,
      "idle_success_rate": count("idle_success") / trials,
      "second_hop_rate": count("second_hop") / trials,
      "mean_delta": mean("delta", "landed"),
      "mean_distance": mean("distance", "landed"),
      "mean_abs_err_x": (data["delta"][landed] - data["target"][landed])
      .abs()
      .mean()
      .item()
      if landed.any()
      else None,
      "mean_apex": mean("apex", "landed"),
      "mean_crouch_drop": mean("crouch_drop", "took_off"),
      "mean_jump_asymmetry": mean("jump_asymmetry", "took_off"),
      "mean_recovery_time": mean("recovery_time", "completed"),
      "mean_standby_torso_tilt_deg": mean(
        "standby_torso_tilt", "commanded", degrees=True
      ),
      "mean_end_torso_tilt_deg": mean("end_torso_tilt", degrees=True),
      "mean_limit_violation": mean("mean_limit_violation"),
      "mean_action_change": mean("mean_action_change"),
      "max_joint_speed": data["max_joint_speed"][m].max().item() if n else None,
      "max_success_distance": data["distance"][m & data["success"]].max().item()
      if count("success")
      else None,
      "failures": {k: v for k, v in fails.items() if v},
    }
    results.append(row)
  return results


def evaluate_seed(args: Args, seed: int, idle: bool) -> list[dict[str, torch.Tensor]]:
  chosen = task_id(args.variant, evaluation=True)
  cfg = load_env_cfg(chosen)
  cfg.seed = seed
  cfg.scene.num_envs = args.trials * (1 if idle else len(args.targets))
  command = cfg.commands["jump"]
  if idle:
    command.fsm.fixed_targets = None
    command.fsm.idle_fraction = 1.0
  else:
    command.fsm.fixed_targets = args.targets
  agent = load_rl_cfg(chosen)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=agent.clip_actions
  )
  try:
    runner = JumpRunner(env, asdict(agent), device=args.device)
    runner.load(
      args.checkpoint_file, load_cfg={"actor": True}, map_location=args.device
    )
    policy = runner.get_inference_policy(device=args.device)
    term = env.unwrapped.command_manager.get_term("jump")
    obs, _ = env.reset()
    term.episode_records.clear()
    seen = torch.zeros(cfg.scene.num_envs, dtype=torch.bool)
    rows = []
    with torch.inference_mode():
      for _ in range(env.unwrapped.max_episode_length + 2):
        obs, _, _, _ = env.step(policy(obs))
        rows.extend(first_episodes(term.episode_records, seen))
        term.episode_records.clear()
        if seen.all():
          break
    return rows
  finally:
    env.close()


def main() -> None:
  args = tyro.cli(Args)
  seeds = args.seeds if args.seeds is not None else (args.seed,)
  if args.trials <= 0 or not seeds or not args.targets:
    raise ValueError("Trials must be positive and targets/seeds must be nonempty.")
  if any(not math.isfinite(t) or t < 0 for t in args.targets):
    raise ValueError("Target distances must be finite and nonnegative.")
  modes = [True] if args.idle else [False, True] if args.include_idle else [False]
  report = {
    "checkpoint": args.checkpoint_file,
    "variant": args.variant,
    "seed": args.seed,
    "seeds": list(seeds),
    "evaluations": [],
  }
  for idle in modes:
    all_rows = []
    targets = (0.0,) if idle else args.targets
    for seed in seeds:
      rows = evaluate_seed(args, seed, idle)
      all_rows.extend(rows)
      report["evaluations"].append(
        {
          "seed": seed,
          "mode": "idle" if idle else "jump",
          "results": summarize(rows, targets, args.trials, idle=idle),
        }
      )
    results = summarize(all_rows, targets, args.trials * len(seeds), idle=idle)
    report["idle_results" if idle else "results"] = results
    print(
      f"\n{'mode':>6} {'target':>7} {'n':>5} {'success':>8} {'survived':>9} {'delta':>7}"
    )
    for r in results:
      distance = f"{r['mean_delta']:.3f}" if r["mean_delta"] is not None else "--"
      target = f"{r['target']:.2f}" if r["target"] is not None else "--"
      success = r["idle_success_rate"] if idle else r["success_rate"]
      survived = r["survival_rate"] if idle else r["success_survived_rate"]
      print(
        f"{'idle' if idle else 'jump':>6} {target:>7} {r['finished']:5d} "
        f"{success:8.2f} {survived:9.2f} {distance:>7}"
      )
  ckpt = Path(args.checkpoint_file)
  suffix = "_idle" if args.idle else ""
  out = (
    Path(args.output)
    if args.output
    else ckpt.with_name(f"eval_{ckpt.stem}{suffix}.json")
  )
  out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
