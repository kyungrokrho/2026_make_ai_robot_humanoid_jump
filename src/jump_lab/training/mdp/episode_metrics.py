"""Evaluation-only diagnostics, collected before automatic reset changes the robot.

These measurements never affect rewards, terminations or the distance curriculum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from .jump_command import JumpCommand


class EpisodeMetrics:
  def __init__(self, term: JumpCommand):
    self.term = term
    self.values = {
      name: torch.zeros(term.num_envs, device=term.device)
      for name in (
        "stable_s",
        "recovery_time",
        "end_tilt_sum",
        "end_steps",
        "limit_sum",
        "action_change_sum",
        "max_joint_speed",
        "steps",
      )
    }
    self.hopped = torch.zeros(term.num_envs, device=term.device, dtype=torch.bool)

  def update(self) -> None:
    term = self.term
    f, env = term.fsm, term._env
    inp = f.inp
    assert inp is not None
    v = self.values
    stable = (
      inp.feet_contact.all(dim=1)
      & (inp.lin_vel.norm(dim=1) < f.cfg.stable_lin_vel)
      & (inp.ang_vel.norm(dim=1) < f.cfg.stable_ang_vel)
      & (inp.tilt < f.cfg.stable_tilt)
      & (inp.torso_tilt < f.cfg.stable_tilt)
    )
    v["stable_s"] = torch.where(stable, v["stable_s"] + env.step_dt, 0.0)
    v["recovery_time"] = torch.where(f.success_event, f.t_cmd, v["recovery_time"])
    end = f.t >= env.cfg.episode_length_s - 0.5
    v["end_tilt_sum"] += inp.torso_tilt * end
    v["end_steps"] += end
    clearance = torch.minimum(inp.heel_pos[..., 2], inp.toe_pos[..., 2]).amin(dim=1)
    self.hopped |= (
      f.landed
      & (f.air_ticks >= f.cfg.takeoff_air_ticks)
      & (clearance >= f.cfg.takeoff_clearance)
    )
    data = term.robot.data
    limits = data.soft_joint_pos_limits
    violation = (limits[..., 0] - data.joint_pos).clamp(min=0)
    violation += (data.joint_pos - limits[..., 1]).clamp(min=0)
    v["limit_sum"] += violation.sum(dim=1)
    am = env.action_manager
    v["action_change_sum"] += (am.action - am.prev_action).square().sum(dim=1)
    v["max_joint_speed"] = torch.maximum(
      v["max_joint_speed"], data.joint_vel.abs().amax(dim=1)
    )
    v["steps"] += 1

  def record(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    term = self.term
    f, env = term.fsm, term._env
    v = self.values
    terminated = env.termination_manager.terminated
    full_episode = env.episode_length_buf >= env.max_episode_length
    survived = full_episode & ~terminated
    record = {
      "env_id": ids,
      "target": f.target[ids],
      "commanded": f.commanded[ids],
      "took_off": f.took_off[ids],
      "landed": f.landed[ids],
      "completed": f.completed[ids],
      "success": f.success[ids],
      "success_survived": (f.success & survived)[ids],
      "success_survived_no_prep_step": (f.success & survived & ~f.prep_step_violation)[
        ids
      ],
      "prep_step_violation": f.prep_step_violation[ids],
      "prep_contact_shift": f.prep_contact_shift[ids],
      "terminated": terminated[ids],
      "survived": survived[ids],
      "idle_success": (
        ~f.commanded & survived & (v["stable_s"] >= f.cfg.recovery_stable_s)
      )[ids],
      "second_hop": self.hopped[ids],
      "fail_code": f.fail_code[ids],
      "delta": f.delta[ids],
      "distance": f.distance[ids],
      "err_y": f.err_y[ids],
      "err_yaw": f.err_yaw[ids],
      "peak_vz": f.vz_peak[ids],
      "apex": f.apex[ids],
      "crouch_drop": f.drop_peak[ids],
      "jump_asymmetry": (term.jump_asym_sum / term.jump_asym_steps.clamp(min=1))[ids],
      "standby_torso_tilt": term.standby_tilt[ids],
      "recovery_time": v["recovery_time"][ids],
      "end_torso_tilt": torch.where(
        v["end_steps"] > 0,
        v["end_tilt_sum"] / v["end_steps"].clamp(min=1),
        torch.full_like(v["end_steps"], float("nan")),
      )[ids],
      "mean_limit_violation": (v["limit_sum"] / v["steps"].clamp(min=1))[ids],
      "mean_action_change": (v["action_change_sum"] / v["steps"].clamp(min=1))[ids],
      "max_joint_speed": v["max_joint_speed"][ids],
    }
    # CPU copy before buffers are reset (also safe if evaluating on CPU).
    return {name: value.detach().cpu().clone() for name, value in record.items()}

  def reset(self, ids: torch.Tensor) -> None:
    for value in self.values.values():
      value[ids] = 0
    self.hopped[ids] = False
