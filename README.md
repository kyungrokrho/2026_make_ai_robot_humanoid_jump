# G1 Jump Lab

Unitree G1이 목표 거리로 점프하도록 강화학습을 구성하고, 학습한 policy를 직접 비교하는 실습 프로젝트입니다. **CPU 노트북에서는 바로 play**, **Linux NVIDIA GPU에서는 PPO 학습**을 할 수 있습니다.

ZIP에 코드, 로봇 모델, 네 개의 학습된 ONNX policy가 모두 포함되어 있습니다. 상위 폴더, 원본 연구 레포, 실험 로그, Git submodule은 필요하지 않습니다. 최초 패키지 설치에는 인터넷이 필요하며, 설치 후 포함된 policy의 play에는 네트워크 연결이 필요하지 않습니다.

| 이름 | 학습 구성 | 비교할 내용 |
| --- | --- | --- |
| `jump` | 기본 구성 | 목표 거리 점프와 자세 회복 |
| `jump_no_symmetry` | symmetry reward와 PPO mirror loss 제거 | 목표 도달과 좌우 대칭성의 차이 |
| `jump_no_curriculum` | 시작부터 최종 거리 분포로 학습 | 쉬운 목표에서 시작하는 과정의 역할 |
| `jump_no_pose` | 기본 자세 tracking reward 제거 | 대기 자세와 착지 후 안정성 |

세 ablation은 각각 기본형에서 해당 요소만 제거한 실험입니다. 포함된 policy는 동일 학습 seed로 10,000 iteration 학습한 결과이며, 파일의 마지막 iteration 번호는 `9999`입니다. ONNX에는 actor와 관측 정규화가 포함됩니다. 학습 재개용 optimizer나 critic은 포함하지 않습니다.

## 1. CPU 노트북에서 바로 play

Windows, macOS, Linux에서 Python 3.11을 사용합니다. NVIDIA GPU, CUDA, PyTorch, MjLab은 play에 필요하지 않습니다. 화면은 Chrome/Edge/Firefox 등의 브라우저에서 표시합니다.

[uv 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 `uv`를 설치한 뒤, ZIP을 풀고 **이 README와 pyproject.toml이 있는 폴더**에서 터미널을 엽니다. 설치 후에도 `uv` 명령이 작동하지 않으면 `pip install uv`로 설치한 뒤 다시 실행합니다.

```bash
uv python install 3.11
uv sync --locked --python 3.11
uv run --no-sync jump-play
```

브라우저에서 **http://localhost:8080** 을 엽니다. 가상환경을 수동으로 activate할 필요는 없습니다.

1. 목표 거리 `Target distance (m)`를 정합니다.
2. `Jump`를 누릅니다. 주황색은 목표 위치, 하늘색은 현재 위치입니다.
3. `Reset`으로 같은 초기 자세로 돌아갑니다.
4. `Policy`를 바꾸어 동일한 거리를 비교합니다. Policy를 바꾸면 자동으로 초기 자세로 돌아갑니다.

`Pause`와 `Speed`로 동작을 천천히 살펴볼 수 있습니다. Jump 명령은 어느 단계에서든 새로 입력할 수 있습니다. 넘어지거나 점프에 실패해도 자동으로 reset하지 않으므로, 다시 시작할 때는 Reset을 누르세요. 실행 종료는 터미널에서 `Ctrl+C`입니다.

다른 policy로 바로 시작하려면:

```bash
uv run --no-sync jump-play --variant jump_no_symmetry
uv run --no-sync jump-play --variant jump_no_curriculum
uv run --no-sync jump-play --variant jump_no_pose
```

화면 없이 실행을 점검하려면 아래 명령을 사용합니다. 2초에 1 m 명령을 주고 총 6초를 시뮬레이션합니다. 출력의 `realtime_factor`는 화면 렌더링을 제외한 속도입니다.

```bash
uv run --no-sync jump-play --headless --target 1 --command-time 2 --duration 6
```

## 2. GPU에서 학습

학습은 **Linux + NVIDIA GPU + NVIDIA 드라이버 + Git**이 필요합니다. 이 프로젝트의 학습 설치는 CUDA 13용 PyTorch wheel을 사용하므로, 호환되는 드라이버가 필요합니다. CPU play 환경과 GPU 학습 환경은 각 컴퓨터에서 별도로 설치합니다.

먼저 `nvidia-smi`가 정상 출력되는지 확인하고, 학습 패키지를 추가합니다.

```bash
uv sync --locked --python 3.11 --extra train
```

먼저 작은 실행으로 GPU와 환경 구성을 점검합니다. 이 실행으로 점프를 배우는 것은 아닙니다.

```bash
uv run --no-sync jump-train --gpu-ids '[0]' --env.scene.num-envs 32 --agent.max-iterations 2 --agent.run-name smoke
```

기본형 학습:

```bash
uv run --no-sync jump-train --variant jump --gpu-ids '[0]' --env.scene.num-envs 8192 --agent.seed 42 --agent.max-iterations 10000
```

세 ablation 중 하나를 학습하려면 `--variant`만 변경합니다.

```bash
uv run --no-sync jump-train --variant jump_no_symmetry --gpu-ids '[0]'
uv run --no-sync jump-train --variant jump_no_curriculum --gpu-ids '[0]'
uv run --no-sync jump-train --variant jump_no_pose --gpu-ids '[0]'
```

위 명령은 각각 독립 학습입니다. 기본값은 환경 8192개, seed 42, 10,000 iteration입니다. GPU 메모리가 부족하면 `--env.scene.num-envs 1024`처럼 환경 수를 줄이세요. 환경 수를 바꾸면 iteration당 수집하는 데이터 양도 달라지므로, 같은 iteration 수가 같은 학습량을 뜻하지는 않습니다.

| 결과 | 저장 위치 |
| --- | --- |
| 체크포인트 | `logs/rsl_rl/<variant>/<실행 폴더>/model_<iteration>.pt` |
| 학습 설정·코드 snapshot | 같은 실행 폴더의 `params/` |
| TensorBoard 기록 | 같은 실행 폴더 |

기본 logger는 TensorBoard입니다. W&B 계정 없이 학습할 수 있습니다. W&B를 사용하려면 로그인 후 logger를 바꿉니다.

```bash
uv run --no-sync wandb login
uv run --no-sync jump-train --variant jump --gpu-ids '[0]' --agent.logger wandb
```

학습 환경의 관측·보상·커리큘럼·종료 조건은 [학습 구성](docs/training.md)에 설명되어 있습니다. 학습 시간과 수렴은 GPU, 환경 수, seed에 따라 달라집니다.

## 3. 내가 학습한 체크포인트를 CPU에서 play

GPU 학습 환경에서 원하는 체크포인트를 export합니다. 아래 `실행폴더`는 실제 생성된 폴더 이름으로 바꿉니다.

```bash
uv run --no-sync jump-export --variant jump --checkpoint logs/rsl_rl/jump/실행폴더/model_9999.pt --output exports/my_jump
```

Export는 ONNX policy, 로봇 XML·mesh, 관측·제어 설정을 묶습니다. 실제 rollout 관측 300개에서 PyTorch와 ONNX의 출력이 일치하는지도 검사합니다. 출력 폴더는 새 폴더여야 합니다.

`exports/my_jump/` **폴더 전체를** 노트북으로 복사하고, CPU play 환경에서 실행합니다.

```bash
uv run --no-sync jump-play --policy exports/my_jump
```

중간 체크포인트도 같은 방식으로 export할 수 있습니다. 예를 들어 `model_1000.pt`와 `model_3000.pt`를 각각 다른 폴더로 export해 비교합니다. Export한 폴더를 `policies/` 바로 아래에 두면 화면의 Policy 목록에도 표시됩니다.

학습 평가가 필요하면 GPU 환경에서 실행합니다. 이 평가는 학습의 종료 조건을 적용하며, 종료 조건을 끈 대화형 play와 목적이 다릅니다.

```bash
uv run --no-sync jump-eval --variant jump --checkpoint-file logs/rsl_rl/jump/실행폴더/model_9999.pt --trials 128 --include-idle --output evaluation.json
```

## 4. 코드 읽는 순서

```text
src/jump_lab/
├── variants.py               # jump + 세 ablation 이름
├── training/
│   ├── env_cfg.py            # 기본 학습 환경: 여기부터 읽기
│   ├── ablations.py          # 세 ablation의 차이
│   ├── rl_cfg.py             # PPO와 mirror loss
│   ├── mdp/rewards.py        # 실제 사용하는 10개 reward
│   ├── mdp/observations.py   # actor 103 / critic 116 입력
│   ├── mdp/fsm.py            # 이륙·착지·회복 단계
│   ├── mdp/curriculum.py    # 목표 거리 확장
│   └── scripts/             # 학습, 평가, export 진입점
└── deploy/
    ├── runtime.py            # MuJoCo CPU + ONNX 제어 루프
    └── play.py               # Viser 화면과 CLI
policies/                     # 네 개의 독립 실행 가능한 policy bundle
```

학생이 주로 수정할 파일은 `env_cfg.py`, `mdp/rewards.py`, `ablations.py`, `rl_cfg.py`입니다. CPU play는 학습 코드의 reward와 critic을 실행하지 않습니다. actor에 필요한 관측, 점프 단계, 물리·제어만 실행합니다.

[비교 실습](docs/exercises.md)에는 관찰 순서와 기록 표가 있습니다. 모델 출처와 제3자 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리했습니다.

## 5. 실행이 안 될 때

| 증상 | 확인할 것 |
| --- | --- |
| `uv` 명령을 찾을 수 없음 | uv 설치 후 터미널을 새로 열기 |
| Policy bundle을 찾을 수 없음 | ZIP 전체를 풀었는지, `policies/`가 있는지 확인 |
| 8080 포트가 이미 사용 중 | `jump-play --port 8081`로 실행 후 localhost:8081 접속 |
| 서버에서 실행한 화면을 노트북에서 보고 싶음 | SSH `-L 8080:localhost:8080` 포트 전달 후 노트북에서 localhost:8080 접속 |
| 브라우저 화면이 느림 | 브라우저 하드웨어 가속을 켜고, 다른 탭을 닫거나 Speed를 낮추기 |
| 학습에서 `mjlab`을 찾을 수 없음 | Linux GPU 환경에서 `uv sync --locked --extra train` 실행 |
| 학습 CUDA 메모리 부족 | `--env.scene.num-envs`를 줄이고 다른 GPU 작업 확인 |
| Checkpoint variant 불일치 | 학습할 때 사용한 variant로 export/eval 실행 |

## 6. 테스트와 ZIP 만들기

```bash
uv sync --locked --extra dev
uv run --no-sync pytest -q
uv run --no-sync python tools/package_release.py
```

GPU 학습 테스트까지 실행하려면 첫 명령에 `--extra train`도 추가합니다. CPU 환경에서는 학습 테스트를 건너뜁니다. ZIP은 `dist/jump_lab.zip`으로 생성하며, 가상환경·로그·캐시·개인 학습 결과는 넣지 않습니다. 학습 환경에서 `uv sync`를 다시 실행할 때는 `--extra train`을 함께 지정해야 학습 패키지가 유지됩니다.

검증 범위: Linux의 독립 CPU 환경과 NVIDIA GPU 환경에서 테스트했습니다. Windows와 macOS용 패키지 구성을 제공하지만, 실제 노트북에서의 실행 확인은 별도로 필요합니다.
