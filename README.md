# 🎮 UOS_RL2025: Bomberman Multi-Agent Reinforcement Learning

## 📘 프로젝트 개요

이 저장소는 **봄버맨 환경에서의 Multi-Agent Reinforcement Learning (MARL)** 실험을 위한 코드와 문서를 포함합니다.
**MAPPO (Multi-Agent Proximal Policy Optimization)** 를 중심으로, 다양한 에이전트 비교 및 성능 향상을 목표로 합니다.

---

## 📁 프로젝트 구조

```plaintext
/Users/joe/proj/bomberman_rl/
├── benchmark.py                    # 베이스라인 측정 스크립트
├── MAPPO_DESIGN.md                 # MAPPO 설계 문서 (15페이지)
├── QUICK_START.md                  # 빠른 시작 가이드
├── RESULTS_SUMMARY.md              # 중간발표 결과 요약
├── TRAINING_GUIDE.md               # 학습 실행 상세 가이드
├── ANALYZE_GUIDE.md                # 결과 분석 가이드
├── test_import.py                  # 환경 테스트
├── run_agent.sh                    # 학습 실행 스크립트
├── monitor_training.sh             # 모니터링 스크립트
├── analyze_results.py              # 결과 분석 (이미 존재)
│
└── agent_code/
    ├── dqn_agent/                  # DQN 프로토타입
    │   ├── callbacks.py
    │   └── train.py
    │
    └── mappo_agent/                # MAPPO 프로토타입
        ├── config.py
        ├── features.py
        ├── models.py
        ├── callbacks.py
        ├── train.py
        ├── checkpoints/
        └── logs/
```

---

## 🚀 빠른 시작 (Quick Start)

### 1️⃣ 환경 확인

```bash
python test_import.py
```

> 모든 항목이 ✓ 나오면 OK!

### 2️⃣ 첫 학습 실행 (약 5분)

```bash
./run_agent.sh mappo_agent --rounds 100
```

### 3️⃣ 결과 확인

```bash
python benchmark.py --agents mappo_agent rule_based_agent --rounds 20
```

---

## 🎯 여러 학습 모드 비교

```bash
# DQN 학습
./run_agent.sh dqn_agent --rounds 500 --scenario coin-heaven

# MAPPO 일반 학습
./run_agent.sh mappo_agent --rounds 500 --scenario coin-heaven

# MAPPO Self-Play
./run_agent.sh mappo_agent --self-play --rounds 500 --scenario coin-heaven

# MAPPO vs Rule-based
./run_agent.sh mappo_agent --vs-rule --rounds 500 --scenario classic

# 성능 비교
python benchmark.py --agents dqn_agent mappo_agent rule_based_agent --rounds 50

# 결과 시각화
python analyze_results.py
```

---

## 🌙 밤새 학습 (Background Training)

```bash
# 백그라운드 실행
nohup ./run_agent.sh mappo_agent --self-play --rounds 5000 > training.log 2>&1 &

# 모니터링
./monitor_training.sh mappo_agent

# 로그 실시간 확인
tail -f training.log
```

---

## 📊 베이스라인 결과

| Agent            | Avg Score | Coins | Kills | Suicides | Survival% |
| ---------------- | --------- | ----- | ----- | -------- | --------- |
| rule_based_agent | **8.76**  | 438   | 0     | 0        | 100.0%    |
| random_agent     | 0.00      | 0     | 0     | 50       | 0.0%      |

> 🎯 **목표:** 8.76점 돌파!

---

## ⚙️ 실행 옵션 정리

| 옵션                 | 설명             | 예시                   |
| ------------------ | -------------- | -------------------- |
| `--rounds N`       | 학습 라운드 수       | `--rounds 2000`      |
| `--scenario SCENE` | 시나리오 선택        | `--scenario classic` |
| `--self-play`      | 4인 self-play   | `--self-play`        |
| `--vs-rule`        | rule_based와 대결 | `--vs-rule`          |
| `--gui`            | GUI 표시         | `--gui`              |
| `--eval`           | 평가 모드 (학습X)    | `--eval`             |

---

## 🗺 시나리오 선택

| 시나리오          | 난이도   | 목적         |
| ------------- | ----- | ---------- |
| `coin-heaven` | 쉬움    | 이동 + 코인 수집 |
| `loot-crate`  | 중간    | 폭탄 사용 연습   |
| `classic`     | 어려움   | 전투 포함      |
| `empty`       | 최고난이도 | 순수 전투 중심   |
