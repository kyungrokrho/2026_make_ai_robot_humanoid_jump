"""Jump command term: owns the state machine and feeds it simulator state."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from jump_lab.variants import Variant

from .episode_metrics import EpisodeMetrics
from .fsm import (
  FAIL_NAMES,
  PHASE_FLIGHT,
  PHASE_PREP,
  PHASE_STANDBY,
  JumpFsm,
  JumpFsmCfg,
  JumpInputs,
)

if TYPE_CHECKING:
  import viser
  from mjlab.envs import ManagerBasedRlEnv

# Sole reference points in the ankle roll link frame, from the foot capsules of the
# G1 MJCF: capsule ends at x = -0.054 and 0.132, z = -0.025, radius 0.01.
HEEL_OFFSET = (-0.064, 0.0, -0.035)
TOE_OFFSET = (0.142, 0.0, -0.035)
FOOT_LENGTH = TOE_OFFSET[0] - HEEL_OFFSET[0]


def mirror_joints(names: tuple[str, ...]) -> tuple[list[int], list[float]]:
  """Left-right mirror of a joint vector: ``q_mirrored = sign * q[index]``.

  Left and right joints swap. Roll and yaw joints change sign, pitch joints (and
  the knee and elbow) do not.
  """
  swap = {"left": "right", "right": "left"}
  index, sign = [], []
  for name in names:
    side = name.split("_", 1)[0]
    other = name.replace(side, swap[side], 1) if side in swap else name
    index.append(names.index(other))
    sign.append(-1.0 if ("_roll_" in name or "_yaw_" in name) else 1.0)
  return index, sign


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  variant: Variant = "jump"
  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
  contact_force_threshold: float = 5.0
  """Ground force in newtons above which a foot counts as in contact."""
  fsm: JumpFsmCfg = field(default_factory=JumpFsmCfg)
  record_episodes: bool = False
  """Keep per-episode results in ``episode_records`` (evaluation)."""
  # The base class resamples on a timer. Jump commands change only at reset.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
    return JumpCommand(self, env)


class JumpCommand(CommandTerm):
  cfg: JumpCommandCfg

  def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
    foot_ids, foot_names = self.robot.find_bodies(
      cfg.foot_body_names, preserve_order=True
    )
    assert list(foot_names) == list(cfg.foot_body_names), foot_names
    self.foot_ids = torch.tensor(foot_ids, device=self.device)
    self.heel_offset = torch.tensor(HEEL_OFFSET, device=self.device)
    self.toe_offset = torch.tensor(TOE_OFFSET, device=self.device)

    mass = self.robot.data.model.body_mass[:, self.robot.indexing.body_ids]
    self.body_mass = mass / mass.sum(dim=1, keepdim=True)  # [B or 1, nbody]

    index, sign = mirror_joints(tuple(self.robot.joint_names))
    self.mirror_index = torch.tensor(index, device=self.device)
    self.mirror_sign = torch.tensor(sign, device=self.device)

    self.torso_id = self.robot.find_bodies("torso_link")[0][0]
    self.down = torch.tensor([0.0, 0.0, -1.0], device=self.device)
    self.fsm = JumpFsm(cfg.fsm, self.num_envs, self.device)
    self._last_update = -1
    self.standby_tilt = torch.zeros(self.num_envs, device=self.device)
    self.standby_pose_err = torch.zeros(self.num_envs, device=self.device)
    # Left-right asymmetry summed over the prep and flight steps of the episode.
    self.jump_asym_sum = torch.zeros(self.num_envs, device=self.device)
    self.jump_asym_steps = torch.zeros(self.num_envs, device=self.device)

    # Curriculum statistics over the top target band, consumed by the curriculum term.
    self.band_count = 0
    self.band_success = 0
    self.episode_records: list[dict[str, torch.Tensor]] = []
    self.episode_metrics = EpisodeMetrics(self) if cfg.record_episodes else None

  @property
  def command(self) -> torch.Tensor:
    return self.fsm.command

  # State machine update.

  def ensure_updated(self) -> None:
    """Advance the state machine once per env step.

    Called by every termination and reward term, so the first one to run after the
    physics step updates the shared state and the rest read the same events.
    """
    step = self._env.common_step_counter
    if step == self._last_update:
      return
    self._last_update = step
    inp = self.gather_inputs()
    # Posture at the moment the command is accepted, for logging.
    standby = self.fsm.phase == PHASE_STANDBY
    self.standby_tilt = torch.where(standby, inp.torso_tilt, self.standby_tilt)
    self.standby_pose_err = torch.where(standby, inp.pose_err, self.standby_pose_err)
    self.fsm.step(inp, self._env.step_dt)
    jumping = (self.fsm.phase == PHASE_PREP) | (self.fsm.phase == PHASE_FLIGHT)
    self.jump_asym_sum += inp.asymmetry * jumping
    self.jump_asym_steps += jumping
    if self.episode_metrics is not None:
      self.episode_metrics.update()

  def gather_inputs(self) -> JumpInputs:
    data = self.robot.data
    foot_pos = data.body_link_pos_w[:, self.foot_ids]
    foot_quat = data.body_link_quat_w[:, self.foot_ids]
    heel = foot_pos + quat_apply(foot_quat, self.heel_offset.expand_as(foot_pos))
    toe = foot_pos + quat_apply(foot_quat, self.toe_offset.expand_as(foot_pos))

    cdata = self.sensor.data
    # Any substep above the threshold counts, so contacts shorter than a policy tick
    # are not missed.
    if cdata.force_history is not None:
      force = cdata.force_history.norm(dim=-1).max(dim=-1).values
    else:
      assert cdata.force is not None
      force = cdata.force.norm(dim=-1)
    contact = force > self.cfg.contact_force_threshold

    w = self.body_mass.unsqueeze(-1)
    com_pos = (data.body_com_pos_w * w).sum(dim=1)
    com_vel = (data.body_com_lin_vel_w * w).sum(dim=1)
    gravity_z = data.projected_gravity_b[:, 2]
    torso_quat = data.body_link_quat_w[:, self.torso_id]
    torso_gz = quat_apply_inverse(torso_quat, self.down.expand(len(torso_quat), 3))[
      :, 2
    ]
    pose_err = (data.joint_pos - data.default_joint_pos).square().sum(dim=1)
    mirrored = data.joint_pos[:, self.mirror_index] * self.mirror_sign
    # Every left-right pair appears twice in the sum, hence the half.
    asymmetry = 0.5 * (data.joint_pos - mirrored).square().sum(dim=1)
    return JumpInputs(
      feet_contact=contact,
      foot_pos=foot_pos,
      heel_pos=heel,
      toe_pos=toe,
      pelvis_pos=data.root_link_pos_w,
      yaw=data.heading_w,
      lin_vel=data.root_link_lin_vel_w,
      ang_vel=data.root_link_ang_vel_w,
      tilt=torch.acos(torch.clamp(-gravity_z, -1.0, 1.0)),
      torso_tilt=torch.acos(torch.clamp(-torso_gz, -1.0, 1.0)),
      pose_err=pose_err,
      asymmetry=asymmetry,
      com_pos=com_pos,
      com_vel=com_vel,
    )

  # CommandTerm interface.

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    if self.episode_metrics is not None:
      # Include failures before a command and idle episodes. The evaluator keeps
      # the FIRST episode of each env, not the first commanded/successful retry.
      finished = env_ids[self.fsm.t[env_ids] > 0]
      if len(finished) > 0:
        self.episode_records.append(self.episode_metrics.record(finished))
      self.episode_metrics.reset(env_ids)
    extras = self._episode_stats(env_ids)
    if len(env_ids) > 0:
      extras["pre_command_failure_rate"] = (
        (
          ~self.fsm.commanded[env_ids]
          & self._env.termination_manager.terminated[env_ids]
        )
        .float()
        .mean()
        .item()
      )
    self.jump_asym_sum[env_ids] = 0.0
    self.jump_asym_steps[env_ids] = 0.0
    super().reset(env_ids)
    return extras

  def _episode_stats(self, env_ids: torch.Tensor) -> dict[str, float]:
    f = self.fsm
    cmd = f.commanded[env_ids]
    n = int(cmd.sum())
    if n == 0:
      return {}
    ids = env_ids[cmd]

    def rate(x: torch.Tensor) -> float:
      return x[ids].float().mean().item()

    stats = {
      "takeoff_rate": rate(f.took_off),
      "landing_rate": rate(f.landed),
      "recovery_rate": rate(f.completed),
      "success_rate": rate(f.success),
      "peak_vz": f.vz_peak[ids].mean().item(),
      "standby_torso_tilt_deg": math.degrees(self.standby_tilt[ids].mean().item()),
      "standby_pose_err": self.standby_pose_err[ids].mean().item(),
      "recovery_stable_s": f.stable_time[ids[f.landed[ids]]].mean().item()
      if f.landed[ids].any()
      else 0.0,
      "delta_max": f.delta_max,
      "prep_step_violation_rate": rate(f.prep_step_violation),
      "prep_contact_shift": f.prep_contact_shift[ids].mean().item(),
    }
    for code, name in FAIL_NAMES.items():
      stats[f"fail_{name}"] = rate(f.fail_code == code)
    landed = ids[f.landed[ids]]
    if len(landed) > 0:
      stats["landed_delta"] = f.delta[landed].mean().item()
      stats["landed_apex"] = f.apex[landed].mean().item()
      stats["landed_abs_err_x"] = f.err_x[landed].abs().mean().item()
      stats["landed_abs_err_y"] = f.err_y[landed].abs().mean().item()
      stats["landed_abs_err_yaw"] = f.err_yaw[landed].abs().mean().item()
    took = ids[f.took_off[ids]]
    if len(took) > 0:
      stats["jump_asymmetry"] = (
        (self.jump_asym_sum[took] / self.jump_asym_steps[took].clamp(min=1))
        .mean()
        .item()
      )
      stats["crouch_drop"] = f.drop_peak[took].mean().item()
      stats["prep_time"] = f.t_takeoff[took].mean().item()

    band = ids[f.target[ids] >= f.delta_max - f.cfg.band_width]
    self.band_count += len(band)
    self.band_success += int(f.success[band].sum())

    return stats

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.fsm.reset(env_ids)

  def _update_metrics(self) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids

  # Play button.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del on_change, request_action
    with server.gui.add_folder(name.capitalize()):
      target = server.gui.add_slider(
        "Target foot displacement (m)", min=0.0, max=2.0, step=0.05, initial_value=0.0
      )
      all_envs = server.gui.add_checkbox("All envs", initial_value=False)
      button = server.gui.add_button("Jump")

      @button.on_click
      def _(_) -> None:
        ids = slice(None) if all_envs.value else get_env_idx()
        self.fsm.pending_target[ids] = float(target.value)
        self.fsm.request[ids] = True
