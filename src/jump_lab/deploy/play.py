"""Play a trained jump policy on CPU, with a browser-based Viser interface."""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

import numpy as np

from jump_lab.deploy.runtime import PHASE_NAMES, Playback
from jump_lab.variants import VARIANTS

ROOT = Path(__file__).resolve().parents[3]


def headless(player, duration, command_time, target):
  if duration <= 0 or command_time < 0 or command_time >= duration:
    raise ValueError("Require duration > command-time >= 0")
  started = time.perf_counter()
  rows = []
  for i in range(round(duration / player.dt)):
    if i == max(0, round(command_time / player.dt) - 1):
      player.request(target)
    player.step()
    _, _, distance = player.markers()
    rows.append(
      {
        "time": player.steps * player.dt,
        "phase": player.state.phase,
        "pelvis_height": float(player.data.xpos[player.pelvis_id, 2]),
        "displacement": distance,
      }
    )
  wall = time.perf_counter() - started
  return {
    "variant": player.config["variant"],
    "duration": len(rows) * player.dt,
    "wall_seconds": wall,
    "realtime_factor": len(rows) * player.dt / wall,
    "took_off": player.state.took_off,
    "landed": player.state.landed,
    "landing_distance": player.state.landing_distance,
    "frames": rows,
  }


def serve(player, policy_root, host, port):
  import viser
  from mjviser import ViserMujocoScene

  server = viser.ViserServer(host=host, port=port)
  server.scene.set_up_direction("+z")
  scene = ViserMujocoScene(server, player.model, num_envs=1)
  commands = queue.SimpleQueue()
  bundles = {
    path.name: path
    for path in sorted(policy_root.iterdir())
    if path.is_dir() and (path / "policy.onnx").exists()
  }
  bundles[player.bundle.name] = player.bundle
  with server.gui.add_folder("Jump Lab"):
    variant = server.gui.add_dropdown(
      "Policy", options=list(bundles), initial_value=player.bundle.name
    )
    target = server.gui.add_slider(
      "Target distance (m)", min=0, max=2, step=0.1, initial_value=1.0
    )
    jump = server.gui.add_button("Jump")
    reset = server.gui.add_button("Reset")
    paused = server.gui.add_checkbox("Pause", initial_value=False)
    speed = server.gui.add_dropdown(
      "Speed", options=("0.25x", "0.5x", "1x"), initial_value="1x"
    )
    status = server.gui.add_markdown("")
    server.gui.add_markdown(
      "**Orange:** target · **Cyan:** current position\n\nChanging policy resets the robot. Jump can be pressed at any time."
    )

  @jump.on_click
  def _jump(_):
    commands.put(("jump", target.value))

  @reset.on_click
  def _reset(_):
    commands.put(("reset", None))

  @variant.on_update
  def _variant(_):
    commands.put(("policy", bundles[variant.value]))

  @server.on_client_connect
  def _camera(client):
    client.camera.position = (2.2, -3.0, 1.7)
    client.camera.look_at = (0.6, 0, 0.6)

  goal_handle = server.scene.add_icosphere("/goal", radius=0.045, color=(255, 150, 30))
  current_handle = server.scene.add_icosphere(
    "/current", radius=0.035, color=(20, 210, 235)
  )
  goal_line = server.scene.add_line_segments(
    "/goal_line",
    points=np.zeros((1, 2, 3), dtype=np.float32),
    colors=(255, 150, 30),
    line_width=4,
  )
  tick = 0
  print(f"Open http://localhost:{port} in your browser. Ctrl+C to stop.", flush=True)
  try:
    while True:
      start = time.perf_counter()
      while not commands.empty():
        command, value = commands.get_nowait()
        if command == "jump":
          player.request(value)
        elif command == "reset":
          player.reset()
        elif command == "policy":
          # Exported bundles use the same robot topology. Check before reuse.
          replacement = Playback(value)
          if (
            replacement.model.nbody != player.model.nbody
            or replacement.model.ngeom != player.model.ngeom
          ):
            raise ValueError(
              "Selected policy uses a different robot model; launch it separately"
            )
          player = replacement
      if not paused.value:
        player.step()
      scene.update_from_mjdata(player.data)
      goal, current, distance = player.markers(target.value)
      goal_handle.position = (*goal, 0.025)
      current_handle.position = (*current, 0.02)
      lateral = player.state.rotation[:, 1] * 0.3
      goal_line.points = np.array(
        [[(*(goal - lateral), 0.012), (*(goal + lateral), 0.012)]], dtype=np.float32
      )
      if tick % 10 == 0:
        status.content = f"**{player.config['variant']}** · {PHASE_NAMES[player.state.phase]}\n\nTime: {player.steps * player.dt:.2f} s · Distance: {distance:.2f} m"
      tick += 1
      time.sleep(
        max(0, player.dt / float(speed.value[:-1]) - (time.perf_counter() - start))
      )
  except KeyboardInterrupt:
    pass
  finally:
    server.stop()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--variant", choices=VARIANTS, default="jump")
  parser.add_argument(
    "--policy", type=Path, help="An exported bundle directory; overrides --variant"
  )
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument(
    "--headless", action="store_true", help="Run without a browser and print a summary"
  )
  parser.add_argument("--duration", type=float, default=6.0)
  parser.add_argument("--command-time", type=float, default=2.0)
  parser.add_argument("--target", type=float, default=1.0)
  parser.add_argument("--output", type=Path, help="Write headless trajectory as JSON")
  args = parser.parse_args()
  policy_root = ROOT / "policies"
  bundle = args.policy or policy_root / args.variant
  if not (bundle / "config.json").exists():
    parser.error(
      f"Policy bundle not found: {bundle}. Extract the complete project ZIP or pass --policy."
    )
  player = Playback(bundle)
  if args.headless:
    result = headless(player, args.duration, args.command_time, args.target)
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
      json.dumps(
        {key: value for key, value in result.items() if key != "frames"}, indent=2
      )
    )
  else:
    serve(player, policy_root, args.host, args.port)


if __name__ == "__main__":
  main()
