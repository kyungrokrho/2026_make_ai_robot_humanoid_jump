# 점프 policy 비교 실습

## 1. 같은 명령, 다른 동작 — 10분

각 policy에서 Reset한 뒤 약 2초 기다리고, 목표 거리 1 m로 Jump를 누릅니다. `Speed = 0.5x`로 준비·비행·착지를 구분해 봅니다. Policy를 바꿀 때 초기 자세도 함께 reset됩니다.

| Policy | 명령 전 서 있는 자세 | 준비 동작 | 좌우 대칭 | 실제 이동 거리 | 착지 후 자세 |
| --- | --- | --- | --- | --- | --- |
| jump | | | | | |
| jump_no_symmetry | | | | | |
| jump_no_curriculum | | | | | |
| jump_no_pose | | | | | |

목표 거리에 도달하는 것과 자연스럽고 안정적인 동작을 만드는 것은 서로 다른 관찰 항목입니다. 현재 위치 표시는 발뒤꿈치 기준 전방 위치입니다. 공중의 골반 위치와 혼동하지 마세요.

## 2. 제거한 설정 찾기 — 10분

먼저 [ablations.py](../src/jump_lab/training/ablations.py)를 읽고, [rl_cfg.py](../src/jump_lab/training/rl_cfg.py)에서 mirror loss 설정을 찾습니다.

| 실험 | 제거한 설정 | 그대로 남은 설정 | 관찰한 결과와 연결해 설명 |
| --- | --- | --- | --- |
| no_symmetry | symmetry reward + mirror loss | 기본 자세, 양발 이륙·착지 | |
| no_curriculum | 거리 상한의 단계적 증가 | 목표 샘플링 비율, reward | |
| no_pose | pose reward | 초기 자세, action offset, still/alive | |

동일한 이름의 보상을 play 중에 끄는 실험이 아닙니다. 서로 다른 학습 구성을 통해 만들어진 policy들을 비교하는 실험입니다.

## 3. 목표 거리를 바꿔 보기 — 10분

각 거리마다 Reset 후 명령을 줍니다. 실패해도 자동으로 reset되지 않으므로, 끝까지 관찰한 뒤 직접 Reset합니다.

| Policy | 0 m | 0.5 m | 1 m | 1.5 m | 2 m |
| --- | --- | --- | --- | --- | --- |
| jump | | | | | |
| jump_no_curriculum | | | | | |

짧은 거리는 수행하지만 긴 거리는 수행하지 못하는지, 혹은 거리 명령 자체에 거의 반응하지 않는지 구분해 기록합니다. 네 policy는 단일 학습 seed의 결과입니다. 이 실험만으로 어떤 설정이 모든 학습에서 반드시 필요하다고 결론내리지는 않습니다.

## 4. 학습 구성을 한 가지 바꿔 보기 — 10~20분 + GPU 학습 시간

[env_cfg.py](../src/jump_lab/training/env_cfg.py)와 [rewards.py](../src/jump_lab/training/mdp/rewards.py)를 확인합니다. 변경 전 예상 결과를 적고, 한 번에 한 가지 설정만 바꿉니다.

| 선택 예시 | 예상 변화 | 확인할 항목 |
| --- | --- | --- |
| pose weight 변경 | 대기·회복 자세의 중요도 변화 | 명령 전 자세, 착지 후 안정성 |
| symmetry weight 변경 | 좌우 대칭 선호 변화 | 거리 정확도와 좌우 차이 |
| curriculum 시작 거리 변경 | 초기 탐색 난이도 변화 | 짧은 거리 성공과 거리 확장 |

새 학습 결과는 기존 `policies/`를 덮어쓰지 말고 별도 폴더로 export합니다. 가능하면 중간 체크포인트와 최종 체크포인트를 함께 비교합니다. 40~50분 수업에서는 제공된 policy 비교를 먼저 하고, 긴 GPU 학습은 별도 실습으로 진행할 수 있습니다.
