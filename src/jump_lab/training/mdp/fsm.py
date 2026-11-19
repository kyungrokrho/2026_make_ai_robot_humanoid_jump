"""Per-env jump state machine.

Pure tensor logic with no simulator dependency, so it can be unit tested with
synthetic contact sequences. The command term feeds it one ``JumpInputs`` per
policy tick, after the physics step and before terminations and rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

PHASE_STANDBY = 0
PHASE_PREP = 1
PHASE_FLIGHT = 2
PHASE_RECOVERY = 3
PHASE_DONE = 4
NUM_PHASES = 5

FAIL_NONE = 0
FAIL_PREP_STEP = 1
FAIL_NO_TAKEOFF = 2
FAIL_FLIGHT_TIMEOUT = 3
FAIL_ONE_FOOT_LANDING = 4
FAIL_SECOND_HOP = 5
FAIL_RECOVERY_TIMEOUT = 6
FAIL_NAMES = {
  FAIL_PREP_STEP: "prep_step",
  FAIL_NO_TAKEOFF: "no_takeoff",
  FAIL_FLIGHT_TIMEOUT: "flight_timeout",
  FAIL_ONE_FOOT_LANDING: "one_foot_landing",
  FAIL_SECOND_HOP: "second_hop",
  FAIL_RECOVERY_TIMEOUT: "recovery_timeout",
}



@dataclass
class JumpFsmCfg:
  mode: Literal["auto", "button"] = "auto"
  """auto: accept unconditionally at a sampled time (training, evaluation).
  button: immediately accept a request from any phase (play)."""

  request_time_range: tuple[float, float] = (0.5, 2.0)
  idle_fraction: float = 0.2
  """Fraction of auto-mode episodes that never receive a command."""
  delta_max: float = 0.30
  """Upper bound of the sampled target foot displacement in metres."""
  band_width: float = 0.10
  """Width of the top target band whose success rate drives the curriculum."""
  top_band_fraction: float = 0.5
  """Fraction of targets drawn from the top band instead of the whole range, so
  that the longest distance is practised as much as all shorter ones together."""
  fixed_targets: tuple[float, ...] | None = None
  """Evaluation: env i always gets fixed_targets[i % len]. Disables idle episodes."""

  takeoff_air_ticks: int = 3
  takeoff_clearance: float = 0.03
  takeoff_min_rise: float = 0.03
  """Centre of mass must be this far above its height at acceptance. Rules out
  lifting the feet while the body drops."""
  takeoff_max_tilt: float = math.radians(50.0)
  """A long jump leaves the ground leaning forward: the line from the toes to the
  centre of mass is 25 to 30 degrees off vertical for 1 to 2 m."""
  prep_step_tolerance: float = 0.05
  """Largest toe displacement on the ground before takeoff. Measured at the toe,
  which stays put while the heel rises during push-off."""
  takeoff_timeout_s: float = 1.5
  flight_timeout_s: float = 1.0
  double_support_window_s: float = 0.10
  recovery_timeout_s: float = 3.5

  stable_lin_vel: float = 0.15
  stable_ang_vel: float = 0.5
  stable_tilt: float = math.radians(12.0)
  stable_scale: float = 1.0
  """Multiplier on the velocity thresholds. Training uses 2 to tolerate the
  action noise of the stochastic policy."""
  stable_filter_s: float = 0.0
  """Time constant of a low-pass filter on the velocities used by the stillness
  test. Training uses 0.2 s: unfiltered, the jitter from the action noise fails the
  test until the noise has collapsed, long before the jump itself is any good."""
  recovery_stable_s: float = 0.35

  tol_x: float = 0.10
  tol_x_slope: float = 0.05
  """Distance tolerance is ``tol_x + tol_x_slope * target``."""
  tol_y: float = 0.08
  tol_yaw: float = math.radians(15.0)

  apex_min: float = 0.30
  apex_base: float = 0.10
  apex_slope: float = 0.18
  """Centre-of-mass rise above its height at acceptance that earns the full height
  reward: ``max(apex_min, apex_base + apex_slope * target)``. Straightening the legs
  alone gives about 0.06 m and a push from the standing pose about 0.25 m. The slope
  follows the 45 degree ballistic arc of the distance left after the forward lean
  at takeoff and the reach of the legs at landing."""
  push_speed: float = 2.5
  """Upward-speed reward cap in metres per second."""
  crouch_depth: float = 0.15
  """Drop of the centre of mass before push-off that earns the full crouch reward."""
  crouch_margin: float = 0.02
  """The drop counts while the centre of mass is over the feet, from this far behind
  the heels to this far ahead of the toes. Toppling forward is not a crouch."""


@dataclass
class JumpInputs:
  feet_contact: torch.Tensor  # [B, 2] bool
  foot_pos: torch.Tensor  # [B, 2, 3] ankle roll link position, world
  heel_pos: torch.Tensor  # [B, 2, 3] world
  toe_pos: torch.Tensor  # [B, 2, 3] world
  pelvis_pos: torch.Tensor  # [B, 3] world
  yaw: torch.Tensor  # [B]
  lin_vel: torch.Tensor  # [B, 3] pelvis, world
  ang_vel: torch.Tensor  # [B, 3]
  tilt: torch.Tensor  # [B] pelvis, rad
  torso_tilt: torch.Tensor  # [B] rad
  pose_err: torch.Tensor  # [B] summed squared joint error from the default pose
  asymmetry: torch.Tensor  # [B] summed squared left-right joint difference, rad^2
  com_pos: torch.Tensor  # [B, 3] whole body, world
  com_vel: torch.Tensor  # [B, 3] whole body, world


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


class JumpFsm:
  def __init__(self, cfg: JumpFsmCfg, num_envs: int, device: str | torch.device):
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device
    self.delta_max = float(cfg.delta_max)

    def z(*shape, dtype=torch.float):
      return torch.zeros(num_envs, *shape, device=device, dtype=dtype)

    self.phase = z(dtype=torch.long)
    self.fail_code = z(dtype=torch.long)
    self.t = z()
    self.t_cmd = z()
    self.t_flight = z()
    self.t_takeoff = z()
    self.first_contact_t = z()
    self.request_time = z()
    self.target = z()
    self.pending_target = z()
    self.request = z(dtype=torch.bool)
    self.stable_time = z()
    self.air_ticks = z(dtype=torch.long)

    self.anchor_xy = z(2)
    self.anchor_valid = z(dtype=torch.bool)
    self.yaw0 = z()
    self.origin_xy = z(2)
    self.heel0_x = z()
    self.toe0_x = z()
    self.feet0_y = z()
    self.toe0_xy = z(2, 2)
    self.com0 = z(3)

    self.lin_vel_f = z(3)
    self.ang_vel_f = z(3)
    self.drop_peak = z()
    self.prep_contact_shift = z()
    self.prep_step_violation = z(dtype=torch.bool)
    self.drop_gain = z()
    self.vz_peak = z()
    self.vz_gain = z()
    self.apex = z()
    self.delta = z()
    self.distance = z()
    self.err_y = z()
    self.err_yaw = z()

    self.commanded = z(dtype=torch.bool)
    self.took_off = z(dtype=torch.bool)
    self.landed = z(dtype=torch.bool)
    self.completed = z(dtype=torch.bool)
    self.success = z(dtype=torch.bool)

    self.accept_event = z(dtype=torch.bool)
    self.takeoff_event = z(dtype=torch.bool)
    self.landing_event = z(dtype=torch.bool)
    self.success_event = z(dtype=torch.bool)

    self.inp: JumpInputs | None = None
    self.reset(torch.arange(num_envs, device=device))

  # Frame helpers.

  def to_frame(self, xy_w: torch.Tensor) -> torch.Tensor:
    """World xy [B, ..., 2] to the frame fixed at command acceptance."""
    d = xy_w - self.origin_xy.view(-1, *([1] * (xy_w.dim() - 2)), 2)
    c = torch.cos(self.yaw0).view(-1, *([1] * (xy_w.dim() - 2)))
    s = torch.sin(self.yaw0).view(-1, *([1] * (xy_w.dim() - 2)))
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return torch.stack([x, y], dim=-1)

  @property
  def active(self) -> torch.Tensor:
    return (self.phase >= PHASE_PREP) & (self.phase <= PHASE_RECOVERY)

  @property
  def failed(self) -> torch.Tensor:
    return self.fail_code != FAIL_NONE

  @property
  def err_x(self) -> torch.Tensor:
    return self.delta - self.target

  @property
  def tol_x(self) -> torch.Tensor:
    return self.cfg.tol_x + self.cfg.tol_x_slope * self.target

  @property
  def apex_target(self) -> torch.Tensor:
    cfg = self.cfg
    return (cfg.apex_base + cfg.apex_slope * self.target).clamp(min=cfg.apex_min)

  @property
  def vz_target(self) -> torch.Tensor:
    """Upward takeoff speed that reaches the apex target."""
    return torch.full_like(self.target, self.cfg.push_speed)

  @property
  def command(self) -> torch.Tensor:
    a = self.active.float()
    return torch.stack([a, self.target * a, self.t_cmd * a], dim=-1)

  # Reset.

  def reset(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    cfg = self.cfg
    for buf in (
      self.phase, self.fail_code, self.t, self.t_cmd, self.t_flight, self.t_takeoff,
      self.stable_time, self.air_ticks, self.vz_peak, self.vz_gain, self.apex,
      self.drop_peak, self.drop_gain,
      self.prep_contact_shift,
      self.delta, self.distance, self.err_y, self.err_yaw,
    ):  # fmt: skip
      if buf is not None:
        buf[env_ids] = 0
    for buf in (
      self.request, self.anchor_valid, self.commanded, self.took_off, self.landed,
      self.completed, self.success, self.accept_event, self.takeoff_event,
      self.landing_event, self.success_event,
      self.prep_step_violation,
    ):  # fmt: skip
      buf[env_ids] = False
    self.first_contact_t[env_ids] = -1.0

    lo, hi = cfg.request_time_range
    req = torch.rand(n, device=self.device) * (hi - lo) + lo
    if cfg.fixed_targets is not None:
      table = torch.tensor(cfg.fixed_targets, device=self.device)
      target = table[env_ids % len(table)]
    else:
      target = torch.rand(n, device=self.device) * self.delta_max
      top = torch.rand(n, device=self.device) < cfg.top_band_fraction
      band = min(cfg.band_width, self.delta_max)
      target = torch.where(
        top, self.delta_max - band * (target / self.delta_max), target
      )
      idle = torch.rand(n, device=self.device) < cfg.idle_fraction
      req = torch.where(idle, torch.full_like(req, float("inf")), req)
    self.request_time[env_ids] = req
    self.target[env_ids] = target
    self.pending_target[env_ids] = target

  # Step.

  def step(self, inp: JumpInputs, dt: float) -> None:
    cfg = self.cfg
    self.inp = inp
    self.accept_event[:] = False
    self.takeoff_event[:] = False
    self.landing_event[:] = False
    self.success_event[:] = False
    self.vz_gain[:] = 0.0
    self.drop_gain[:] = 0.0

    ph = self.phase.clone()
    contact = inp.feet_contact
    both = contact.all(dim=1)
    none = ~contact.any(dim=1)
    pelvis_xy = inp.pelvis_pos[:, :2]

    self.t += dt
    fresh = ~self.anchor_valid
    self.anchor_xy[fresh] = pelvis_xy[fresh]
    self.anchor_valid[:] = True

    blend = min(1.0, dt / cfg.stable_filter_s) if cfg.stable_filter_s > 0 else 1.0
    self.lin_vel_f[fresh] = inp.lin_vel[fresh]
    self.ang_vel_f[fresh] = inp.ang_vel[fresh]
    self.lin_vel_f += blend * (inp.lin_vel - self.lin_vel_f)
    self.ang_vel_f += blend * (inp.ang_vel - self.ang_vel_f)
    stable = (
      both
      & (self.lin_vel_f.norm(dim=1) < cfg.stable_lin_vel * cfg.stable_scale)
      & (self.ang_vel_f.norm(dim=1) < cfg.stable_ang_vel * cfg.stable_scale)
      & (inp.tilt < cfg.stable_tilt)
      & (inp.torso_tilt < cfg.stable_tilt)
    )
    self.stable_time = torch.where(
      stable, self.stable_time + dt, torch.zeros_like(self.stable_time)
    )
    self.air_ticks = torch.where(
      none, self.air_ticks + 1, torch.zeros_like(self.air_ticks)
    )

    # Standby: accept a command.
    standby = ph == PHASE_STANDBY
    if cfg.mode == "auto":
      accept = standby & (self.t >= self.request_time)
    else:
      accept = self.request.clone()
    if accept.any():
      self._accept(accept, inp)
      if cfg.mode == "button":
        # A new command replaces the old jump. Do not process its stale phase
        # (flight, recovery or done) during this command tick.
        ph[accept] = PHASE_STANDBY
        self.anchor_xy[accept] = pelvis_xy[accept]

    # Preparation and push-off.
    prep = ph == PHASE_PREP
    self.t_cmd[ph >= PHASE_PREP] += dt
    vz = torch.minimum(inp.com_vel[:, 2], self.vz_target)
    com_x = self.to_frame(inp.com_pos[:, :2])[:, 0]
    balanced = (
      both
      & (com_x > self.heel0_x - cfg.crouch_margin)
      & (com_x < self.toe0_x + cfg.crouch_margin)
    )
    drop = self.com0[:, 2] - inp.com_pos[:, 2]
    deeper = (drop - self.drop_peak).clamp(min=0.0) * (prep & balanced)
    # Only the part below the full-reward depth is paid.
    self.drop_gain = (self.drop_peak + deeper).clamp(
      max=cfg.crouch_depth
    ) - self.drop_peak.clamp(max=cfg.crouch_depth)
    self.drop_peak += deeper
    upright = inp.tilt < cfg.takeoff_max_tilt
    gain = (vz - self.vz_peak).clamp(min=0.0) * (prep & upright)
    self.vz_peak += gain
    self.vz_gain = gain

    sole_z = torch.minimum(inp.heel_pos[..., 2], inp.toe_pos[..., 2])  # [B, 2]
    clearance = sole_z.min(dim=1).values
    takeoff = (
      prep
      & (self.air_ticks >= cfg.takeoff_air_ticks)
      & (clearance >= cfg.takeoff_clearance)
      & (inp.com_pos[:, 2] - self.com0[:, 2] >= cfg.takeoff_min_rise)
      & (inp.com_vel[:, 2] > 0.0)
      & upright
    )
    foot_shift = (inp.toe_pos[..., :2] - self.toe0_xy).norm(dim=-1)  # [B, 2]
    stepped = prep & ((foot_shift > cfg.prep_step_tolerance) & contact).any(dim=1)
    self.prep_step_violation |= stepped & ~takeoff
    self.prep_contact_shift = torch.where(
      prep & ~takeoff,
      torch.maximum(self.prep_contact_shift, (foot_shift * contact).amax(dim=1)),
      self.prep_contact_shift,
    )
    blocked_step = stepped
    self._fail(blocked_step & ~takeoff, FAIL_PREP_STEP)
    self._fail(
      prep & ~takeoff & ~blocked_step & (self.t_cmd > cfg.takeoff_timeout_s),
      FAIL_NO_TAKEOFF,
    )
    if takeoff.any():
      self._takeoff(takeoff, inp)

    # Flight until both feet are down.
    flight = ph == PHASE_FLIGHT
    rise = inp.com_pos[:, 2] - self.com0[:, 2]
    self.apex = torch.where(prep | flight, torch.maximum(self.apex, rise), self.apex)
    self.t_flight[flight] += dt
    touched = flight & contact.any(dim=1) & (self.first_contact_t < 0)
    self.first_contact_t[touched] = self.t_flight[touched]
    landing = flight & both
    waited = self.t_flight - self.first_contact_t
    self._fail(
      flight
      & ~landing
      & (self.first_contact_t >= 0)
      & (waited > cfg.double_support_window_s),
      FAIL_ONE_FOOT_LANDING,
    )
    self._fail(
      flight & ~landing & (self.t_flight > cfg.flight_timeout_s), FAIL_FLIGHT_TIMEOUT
    )
    if landing.any():
      self._land(landing, inp)

    # Recovery until the robot has been still for long enough.
    rec = ph == PHASE_RECOVERY
    # Same airborne test as takeoff, so contact chatter on impact is not a hop.
    hop = (
      rec
      & (self.air_ticks >= cfg.takeoff_air_ticks)
      & (clearance >= cfg.takeoff_clearance)
    )
    blocked_hop = hop
    self._fail(blocked_hop, FAIL_SECOND_HOP)
    done = rec & ~hop & (self.stable_time >= cfg.recovery_stable_s)
    self._fail(
      rec & ~blocked_hop & ~done & (self.t_cmd > cfg.recovery_timeout_s),
      FAIL_RECOVERY_TIMEOUT,
    )
    if done.any():
      self.phase[done] = PHASE_DONE
      self.completed[done] = True
      self.success_event[done] = True
      accurate = (
        (self.err_x.abs() <= self.tol_x)
        & (self.err_y.abs() <= cfg.tol_y)
        & (self.err_yaw.abs() <= cfg.tol_yaw)
      )
      self.success[done] = accurate[done]

    # Play: return to standby so the next request can be accepted.
    if cfg.mode == "button":
      back = (ph == PHASE_DONE) & stable
      self.phase[back] = PHASE_STANDBY
      self.anchor_xy[back] = pelvis_xy[back]
      self.stable_time[back] = 0.0

  def _fail(self, mask: torch.Tensor, code: int) -> None:
    mask = mask & (self.fail_code == FAIL_NONE)
    self.fail_code[mask] = code

  def _accept(self, m: torch.Tensor, inp: JumpInputs) -> None:
    # Play can accept multiple jumps without resetting the environment. Clear the
    # previous landing/records so observations and bonuses describe this jump.
    if self.cfg.mode == "button":
      self.fail_code[m] = FAIL_NONE
      self.stable_time[m] = 0.0
      self.t_flight[m] = 0.0
      self.t_takeoff[m] = 0.0
      self.first_contact_t[m] = -1.0
    self.took_off[m] = False
    self.landed[m] = False
    self.completed[m] = False
    self.success[m] = False
    self.prep_step_violation[m] = False
    self.prep_contact_shift[m] = 0.0
    self.apex[m] = 0.0
    self.delta[m] = 0.0
    self.distance[m] = 0.0
    self.err_y[m] = 0.0
    self.err_yaw[m] = 0.0
    self.accept_event[m] = True
    self.commanded[m] = True
    self.phase[m] = PHASE_PREP
    self.request[m] = False
    self.target[m] = self.pending_target[m]
    self.t_cmd[m] = 0.0
    self.vz_peak[m] = 0.0
    self.air_ticks[m] = 0
    self.yaw0[m] = inp.yaw[m]
    self.origin_xy[m] = inp.pelvis_pos[m, :2]
    heel = self.to_frame(inp.heel_pos[..., :2])
    toe = self.to_frame(inp.toe_pos[..., :2])
    foot = self.to_frame(inp.foot_pos[..., :2])
    self.heel0_x[m] = heel[..., 0].min(dim=1).values[m]
    self.toe0_x[m] = toe[..., 0].max(dim=1).values[m]
    self.feet0_y[m] = foot[..., 1].mean(dim=1)[m]
    self.toe0_xy[m] = inp.toe_pos[m][..., :2]
    self.drop_peak[m] = 0.0
    self.com0[m] = inp.com_pos[m]

  def _takeoff(self, m: torch.Tensor, inp: JumpInputs) -> None:
    self.takeoff_event[m] = True
    self.took_off[m] = True
    self.phase[m] = PHASE_FLIGHT
    self.t_flight[m] = 0.0
    self.t_takeoff[m] = self.t_cmd[m]
    self.first_contact_t[m] = -1.0

  def _land(self, m: torch.Tensor, inp: JumpInputs) -> None:
    self.landing_event[m] = True
    self.landed[m] = True
    self.phase[m] = PHASE_RECOVERY
    self.stable_time[m] = 0.0
    self.air_ticks[m] = 0
    self.anchor_xy[m] = inp.pelvis_pos[m, :2]
    heel_x = self.to_frame(inp.heel_pos[..., :2])[..., 0].min(dim=1).values
    foot_y = self.to_frame(inp.foot_pos[..., :2])[..., 1].mean(dim=1)
    self.delta[m] = (heel_x - self.heel0_x)[m]
    self.distance[m] = (heel_x - self.toe0_x)[m]
    self.err_y[m] = (foot_y - self.feet0_y)[m]
    self.err_yaw[m] = wrap_to_pi(inp.yaw - self.yaw0)[m]
