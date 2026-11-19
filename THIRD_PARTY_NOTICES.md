# Third-party notices

## Robot model and meshes

The G1 scene and STL meshes in `policies/*/robot/` are exported from the Unitree G1 asset in [MjLab](https://github.com/mujocolab/mjlab/tree/f135c1daa0f278bd19e323c2b9f256ca541ae2c5/src/mjlab/asset_zoo/robots/unitree_g1). The scene adds the task's terrain, contact sensors, standing pose, collision settings and native position actuators. Mesh geometry is unchanged.

- MjLab contributors: Apache License 2.0, included as [mjlab-Apache-2.0.txt](licenses/mjlab-Apache-2.0.txt).
- Unitree G1 assets: Unitree Robotics; upstream [MuJoCo Menagerie G1 license](https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_g1/LICENSE), included as [unitree-g1.txt](licenses/unitree-g1.txt).

## Installed dependencies

MuJoCo, ONNX Runtime, NumPy, Viser and mjviser are installed as dependencies; their licenses remain in their respective distributions. GPU training additionally installs MjLab, PyTorch, MuJoCo Warp, RSL-RL and their dependencies. See `pyproject.toml` and `uv.lock` for the pinned distribution sources.

The project includes four exported actor policies from the teaching experiments. They are used for simulated robot playback. Each bundle records its variant, iteration, checkpoint digest and ONNX validation results in `config.json` and `export_validation.json`.
