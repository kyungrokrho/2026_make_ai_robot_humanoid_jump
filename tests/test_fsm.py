"""State machine checks on synthetic contact sequences. No simulator needed."""

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("mjlab")
import torch

from jump_lab.training.mdp.fsm import (
  FAIL_NO_TAKEOFF,
  FAIL_NONE,
  FAIL_ONE_FOOT_LANDING,
  FAIL_PREP_STEP,
  FAIL_RECOVERY_TIMEOUT,
  FAIL_SECOND_HOP,
  PHASE_DONE,
  PHASE_FLIGHT,
  PHASE_PREP,
  PHASE_RECOVERY,
  PHASE_STANDBY,
  JumpFsm,
  JumpFsmCfg,
  JumpInputs,
)

DT = 0.02
HEEL, TOE = -0.064, 0.142


def make_inputs(
  x=0.0,
  z=0.0,
  contact=(True, True),
  yaw=0.0,
  speed=0.0,
  vz=None,
  foot_dx=(0.0, 0.0),
  b=1,
):
  """Robot facing +x (rotated by yaw) with feet 0.2 m apart at pelvis x."""
  c, s = math.cos(yaw), math.sin(yaw)
  if vz is None:  # Rising while airborne, at rest otherwise.
    vz = 0.0 if any(contact) else 0.5

  def pt(fx, fy, fz):
    return [x + c * fx - s * fy, s * fx + c * fy, fz]

  foot = [pt(foot_dx[i], y, z + 0.035) for i, y in enumerate((0.1, -0.1))]
  heel = [pt(foot_dx[i] + HEEL, y, z) for i, y in enumerate((0.1, -0.1))]
  toe = [pt(foot_dx[i] + TOE, y, z) for i, y in enumerate((0.1, -0.1))]

  def t(v):
    return torch.tensor(v, dtype=torch.float).expand(b, *torch.tensor(v).shape).clone()

  return JumpInputs(
    feet_contact=torch.tensor(contact).expand(b, 2).clone(),
    foot_pos=t(foot),
    heel_pos=t(heel),
    toe_pos=t(toe),
    pelvis_pos=t([x, 0.0, 0.76 + z]),
    yaw=t(yaw),
    lin_vel=t([speed, 0.0, 0.0]),
    ang_vel=t([0.0, 0.0, 0.0]),
    tilt=t(0.0),
    torso_tilt=t(0.0),
    pose_err=t(0.0),
    asymmetry=t(0.0),
    com_pos=t([x, 0.0, 0.65 + z]),
    com_vel=t([speed, 0.0, vz]),
  )


def make_fsm(**kw) -> JumpFsm:
  cfg = JumpFsmCfg(request_time_range=(0.1, 0.1), idle_fraction=0.0, **kw)
  fsm = JumpFsm(cfg, 1, "cpu")
  fsm.target[:] = 0.3
  fsm.pending_target[:] = 0.3
  return fsm


def run(fsm, n, **kw):
  for _ in range(n):
    fsm.step(make_inputs(**kw), DT)


def accept(fsm):
  run(fsm, 6)
  assert fsm.phase.item() == PHASE_PREP


def jump(fsm, distance=0.3, air=10):
  """Push off, fly ``distance`` forward and land on both feet."""
  run(fsm, 3, vz=1.0)
  for i in range(air):
    frac = (i + 1) / air
    run(fsm, 1, x=distance * frac, z=0.1, contact=(False, False), speed=1.0)
  run(fsm, 1, x=distance)


def test_normal_jump_succeeds():
  fsm = make_fsm()
  accept(fsm)
  assert fsm.command[0].tolist() == pytest.approx([1.0, 0.3, 0.0])
  run(fsm, 3, vz=1.0)
  assert fsm.vz_peak.item() == pytest.approx(1.0)
  run(fsm, 2, z=0.1, contact=(False, False))
  assert fsm.phase.item() == PHASE_PREP  # Needs three airborne ticks.
  run(fsm, 1, x=0.05, z=0.1, contact=(False, False))
  assert fsm.takeoff_event.item() and fsm.phase.item() == PHASE_FLIGHT
  run(fsm, 5, x=0.2, z=0.1, contact=(False, False))
  run(fsm, 1, x=0.3)
  assert fsm.landing_event.item() and fsm.phase.item() == PHASE_RECOVERY
  assert fsm.delta.item() == pytest.approx(0.3, abs=1e-5)
  assert fsm.distance.item() == pytest.approx(0.3 - (TOE - HEEL), abs=1e-5)
  run(fsm, 17, x=0.3)
  assert fsm.phase.item() == PHASE_RECOVERY
  run(fsm, 1, x=0.3)
  assert fsm.success_event.item() and fsm.phase.item() == PHASE_DONE
  assert fsm.success.item() and fsm.fail_code.item() == FAIL_NONE
  assert fsm.command[0, 0].item() == 0.0


def test_events_last_one_tick():
  fsm = make_fsm()
  accept(fsm)
  jump(fsm)
  assert fsm.landing_event.item()
  run(fsm, 1, x=0.3)
  assert not fsm.landing_event.item() and not fsm.takeoff_event.item()


def test_frame_follows_yaw():
  fsm = make_fsm()
  yaw = 2.0
  run(fsm, 6, yaw=yaw)
  run(fsm, 4, z=0.1, contact=(False, False), yaw=yaw)
  # x here is distance along the heading; the helper rotates everything by yaw.
  for _ in range(3):
    inp = make_inputs(z=0.1, contact=(False, False), yaw=yaw)
    fsm.step(inp, DT)
  landed = make_inputs(yaw=yaw)
  shift = torch.tensor([0.3 * math.cos(yaw), 0.3 * math.sin(yaw), 0.0])
  for name in ("foot_pos", "heel_pos", "toe_pos"):
    setattr(landed, name, getattr(landed, name) + shift)
  landed.pelvis_pos += shift
  fsm.step(landed, DT)
  assert fsm.delta.item() == pytest.approx(0.3, abs=1e-5)
  assert fsm.err_y.item() == pytest.approx(0.0, abs=1e-5)


def test_inaccurate_landing_completes_without_success():
  fsm = make_fsm()
  accept(fsm)
  jump(fsm, distance=0.6)
  run(fsm, 18, x=0.6)
  assert fsm.completed.item() and not fsm.success.item()


def test_walking_is_a_prep_step():
  fsm = make_fsm()
  accept(fsm)
  run(fsm, 2, contact=(False, True), foot_dx=(0.1, 0.0))
  assert fsm.fail_code.item() == FAIL_NONE  # Swing foot in the air is allowed.
  run(fsm, 1, foot_dx=(0.1, 0.0))
  assert fsm.fail_code.item() == FAIL_PREP_STEP


def test_low_hop_does_not_take_off():
  fsm = make_fsm()
  accept(fsm)
  run(fsm, 5, z=0.01, contact=(False, False))
  assert fsm.phase.item() == PHASE_PREP
  run(fsm, 80)
  assert fsm.fail_code.item() == FAIL_NO_TAKEOFF


def test_one_foot_landing_fails():
  fsm = make_fsm()
  accept(fsm)
  run(fsm, 4, z=0.1, contact=(False, False))
  run(fsm, 5, x=0.3, contact=(True, False))
  assert fsm.fail_code.item() == FAIL_NONE
  run(fsm, 2, x=0.3, contact=(True, False))
  assert fsm.fail_code.item() == FAIL_ONE_FOOT_LANDING


def test_staggered_landing_within_window_is_ok():
  fsm = make_fsm()
  accept(fsm)
  run(fsm, 4, z=0.1, contact=(False, False))
  run(fsm, 3, x=0.3, contact=(True, False))
  run(fsm, 1, x=0.3)
  assert fsm.phase.item() == PHASE_RECOVERY and fsm.fail_code.item() == FAIL_NONE


def test_second_hop_fails():
  fsm = make_fsm()
  accept(fsm)
  jump(fsm)
  run(fsm, 3, x=0.4, z=0.1, contact=(False, False))
  assert fsm.fail_code.item() == FAIL_SECOND_HOP


def test_never_still_times_out():
  fsm = make_fsm()
  accept(fsm)
  jump(fsm)
  run(fsm, 180, x=0.3, speed=1.0)
  assert fsm.fail_code.item() == FAIL_RECOVERY_TIMEOUT


def test_stable_scale_relaxes_stillness():
  for scale, expected in ((1.0, False), (2.0, True)):
    fsm = make_fsm(stable_scale=scale)
    accept(fsm)
    jump(fsm)
    run(fsm, 18, x=0.3, speed=0.2)
    assert fsm.completed.item() == expected


def test_idle_episode_never_starts():
  fsm = JumpFsm(JumpFsmCfg(idle_fraction=1.0), 4, "cpu")
  for _ in range(200):
    fsm.step(make_inputs(b=4), DT)
  assert (fsm.phase == PHASE_STANDBY).all() and not fsm.commanded.any()
  assert (fsm.command == 0).all()


def test_button_accepts_while_moving_and_jumps_once_per_request():
  fsm = make_fsm(mode="button")
  run(fsm, 50)
  assert fsm.phase.item() == PHASE_STANDBY  # No request yet.
  fsm.request[:] = True
  fsm.pending_target[:] = 0.5
  run(fsm, 1, speed=1.0)
  assert fsm.phase.item() == PHASE_PREP
  assert fsm.target.item() == pytest.approx(0.5) and not fsm.request.item()
  jump(fsm, distance=0.5)
  run(fsm, 18, x=0.5)
  assert fsm.phase.item() == PHASE_DONE
  run(fsm, 1, x=0.5)
  assert fsm.phase.item() == PHASE_STANDBY
  run(fsm, 50, x=0.5)
  assert fsm.phase.item() == PHASE_STANDBY  # Held: no second jump.
  fsm.request[:] = True
  run(fsm, 11, x=0.5)
  assert fsm.phase.item() == PHASE_PREP  # Second request starts from the new spot.
  assert fsm.origin_xy[0, 0].item() == pytest.approx(0.5)


def test_partial_reset_leaves_other_envs_alone():
  cfg = JumpFsmCfg(request_time_range=(0.1, 0.1), idle_fraction=0.0)
  fsm = JumpFsm(cfg, 2, "cpu")
  for _ in range(6):
    fsm.step(make_inputs(b=2), DT)
  assert (fsm.phase == PHASE_PREP).all()
  fsm.reset(torch.tensor([0]))
  assert fsm.phase.tolist() == [PHASE_STANDBY, PHASE_PREP]
  assert fsm.commanded.tolist() == [False, True]
  assert fsm.t.tolist() == pytest.approx([0.0, 0.12])


def test_fixed_targets_cycle_over_envs():
  cfg = JumpFsmCfg(fixed_targets=(0.0, 0.2, 0.4))
  fsm = JumpFsm(cfg, 6, "cpu")
  assert fsm.target.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
  assert torch.isfinite(fsm.request_time).all()


def test_tucking_feet_while_dropping_is_not_a_takeoff():
  fsm = make_fsm()
  accept(fsm)
  for _ in range(6):
    inp = make_inputs(z=0.1, contact=(False, False))
    inp.com_pos[:, 2] = 0.60
    fsm.step(inp, DT)
  assert fsm.phase.item() == PHASE_PREP


def test_tilted_takeoff_is_rejected():
  fsm = make_fsm()
  accept(fsm)
  for _ in range(6):
    inp = make_inputs(z=0.1, contact=(False, False))
    inp.tilt[:] = 1.0
    fsm.step(inp, DT)
  assert fsm.phase.item() == PHASE_PREP


def test_heel_rise_is_not_a_step_but_sliding_toes_are():
  fsm = make_fsm()
  accept(fsm)
  inp = make_inputs()
  inp.foot_pos[..., 0] += 0.08  # Ankle swings forward over planted toes.
  inp.heel_pos[..., 2] += 0.10
  fsm.step(inp, DT)
  assert fsm.fail_code.item() == FAIL_NONE
  inp = make_inputs()
  inp.toe_pos[:, 0, 0] += 0.06
  fsm.step(inp, DT)
  assert fsm.fail_code.item() == FAIL_PREP_STEP


def test_crouch_depth_is_recorded_on_both_feet():
  fsm = make_fsm()
  accept(fsm)
  for dz, contact in ((-0.10, (True, True)), (-0.20, (False, False))):
    inp = make_inputs(contact=contact)
    inp.com_pos[:, 2] += dz
    fsm.step(inp, DT)
  assert fsm.drop_peak.item() == pytest.approx(0.10)


def test_crouch_gain_sums_to_the_full_depth_and_ignores_toppling():
  fsm = make_fsm()
  accept(fsm)
  total = 0.0
  for dz in (-0.05, -0.03, -0.12, -0.25):
    inp = make_inputs()
    inp.com_pos[:, 2] += dz
    fsm.step(inp, DT)
    total += fsm.drop_gain.item()
  assert total == pytest.approx(fsm.cfg.crouch_depth)
  assert fsm.drop_peak.item() == pytest.approx(0.25)

  fsm = make_fsm()
  accept(fsm)
  inp = make_inputs()
  inp.com_pos[:, 0] += TOE + 0.05  # Centre of mass ahead of the toes.
  inp.com_pos[:, 2] -= 0.10
  fsm.step(inp, DT)
  assert fsm.drop_gain.item() == 0.0 and fsm.drop_peak.item() == 0.0


def test_targets_scale_height_and_tolerance():
  fsm = make_fsm()
  fsm.target[:] = 2.0
  assert fsm.apex_target.item() == pytest.approx(0.46)
  assert fsm.tol_x.item() == pytest.approx(0.20)
  fsm.target[:] = 0.2
  assert fsm.apex_target.item() == pytest.approx(0.30)
  assert fsm.vz_target.item() == pytest.approx(2.5)


def test_filter_ignores_velocity_jitter():
  for tau, expected in ((0.0, False), (0.2, True)):
    fsm = make_fsm(stable_filter_s=tau)
    accept(fsm)
    jump(fsm)
    for i in range(60):
      inp = make_inputs(x=0.3)
      inp.lin_vel[:, 0] = 0.3 if i % 2 else -0.3
      fsm.step(inp, DT)
    assert fsm.completed.item() is expected


def test_half_of_the_targets_come_from_the_top_band():
  fsm = JumpFsm(JumpFsmCfg(delta_max=1.0, idle_fraction=0.0), 4000, "cpu")
  assert fsm.target.max().item() <= 1.0 and fsm.target.min().item() >= 0.0
  top = (fsm.target >= 0.9).float().mean().item()
  assert top == pytest.approx(0.55, abs=0.04)


@pytest.mark.parametrize(
  "phase", [PHASE_STANDBY, PHASE_PREP, PHASE_FLIGHT, PHASE_RECOVERY, PHASE_DONE]
)
def test_button_accepts_from_any_phase_without_pose_gate(phase):
  fsm = make_fsm(mode="button")
  fsm.phase[:] = phase
  fsm.request[:] = True
  fsm.pending_target[:] = 0.7
  fsm.t_flight[:] = 10.0
  fsm.fail_code[:] = FAIL_SECOND_HOP
  fsm.landed[:] = True
  fsm.completed[:] = True
  inp = make_inputs(x=0.8, speed=1.0)
  inp.pose_err[:] = 6.0
  inp.torso_tilt[:] = 1.2
  fsm.step(inp, DT)
  assert fsm.accept_event.item()
  assert fsm.phase.item() == PHASE_PREP
  assert not fsm.request.item()
  assert fsm.fail_code.item() == FAIL_NONE
  assert not fsm.landed.item() and not fsm.completed.item()
  assert not fsm.landing_event.item() and not fsm.success_event.item()
  assert fsm.t_cmd.item() == 0.0 and fsm.t_flight.item() == 0.0
  assert fsm.target.item() == pytest.approx(0.7)
  assert fsm.origin_xy[0, 0].item() == pytest.approx(0.8)


def test_leaning_torso_blocks_recovery():
  fsm = make_fsm()
  accept(fsm)
  jump(fsm)
  for _ in range(30):
    inp = make_inputs(x=0.3)
    inp.torso_tilt[:] = 0.4
    fsm.step(inp, DT)
  assert fsm.phase.item() == PHASE_RECOVERY
  run(fsm, 18, x=0.3)
  assert fsm.phase.item() == PHASE_DONE


def test_apex_tracks_highest_rise_until_landing():
  fsm = make_fsm()
  accept(fsm)
  run(fsm, 4, z=0.1, contact=(False, False))
  run(fsm, 3, z=0.22, contact=(False, False))
  run(fsm, 2, z=0.05, contact=(False, False))
  run(fsm, 1, x=0.3)
  assert fsm.apex.item() == pytest.approx(0.22, abs=1e-5)
  run(fsm, 5, x=0.3, z=0.5, contact=(True, True))
  assert fsm.apex.item() == pytest.approx(0.22, abs=1e-5)
