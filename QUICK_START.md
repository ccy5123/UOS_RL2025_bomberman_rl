# Bomberman RL - Quick Start Guide

## 📦 4개 구현 완료!

### 1️⃣ 베이스라인 성능 측정 스크립트 ✅

**파일**: `benchmark.py`

**사용법**:
```bash
# 전체 벤치마크 실행
python benchmark.py

# 빠른 벤치마크 (적은 라운드)
python benchmark.py --quick

# 특정 에이전트 비교
python benchmark.py --agents rule_based_agent random_agent --rounds 100
```

**결과**:
- 각 에이전트의 평균 점수, 코인 수집, 킬 수, 자살률, 생존율
- 1v1 매치업 결과
- 시나리오별 성능

---

### 2️⃣ MAPPO 구조 설계 문서 ✅

**파일**: `MAPPO_DESIGN.md`

**내용**:
- CTDE (Centralized Training, Decentralized Execution) 아키텍처
- Actor (Policy Network) + Centralized Critic 설계
- State representation (7-channel CNN)
- Global state for critic
- PPO 업데이트 알고리즘
- RND (Random Network Distillation) 탐험
- Self-play league 시스템
- Reward shaping 전략
- 8주 로드맵

---

### 3️⃣ DQN 프로토타입 ✅

**폴더**: `agent_code/dqn_agent/`

**파일 구조**:
```
agent_code/dqn_agent/
├── callbacks.py    # DQN 모델, act() 함수
└── train.py        # Experience replay, 학습 로직
```

**특징**:
- 7-channel CNN state representation
- Experience replay buffer
- Target network (stable learning)
- Rule-based guided exploration (ε-greedy)
- Reward shaping

**학습 시작**:
```bash
# coin-heaven에서 학습 시작 (쉬운 난이도)
python main.py play --my-agent dqn_agent --train 1 --no-gui --n-rounds 1000 --scenario coin-heaven

# classic 모드 학습
python main.py play --my-agent dqn_agent --train 1 --no-gui --n-rounds 2000 --scenario classic

# 평가 (GUI)
python main.py play --my-agent dqn_agent --n-rounds 5
```

---

### 4️⃣ MAPPO 프로토타입 ✅

**폴더**: `agent_code/mappo_agent/`

**파일 구조**:
```
agent_code/mappo_agent/
├── config.py       # Hyperparameters
├── features.py     # State extraction (local obs + global state)
├── models.py       # PolicyNetwork, CentralizedValueNetwork, RND
├── callbacks.py    # Agent entry points (setup, act)
├── train.py        # PPO update, GAE, reward shaping
├── checkpoints/    # Saved models
└── logs/           # Training logs
```

**특징**:
- **Centralized Critic**: 모든 에이전트 정보를 사용한 가치 평가
- **Decentralized Actor**: 각 에이전트가 독립적으로 행동 선택
- **Parameter Sharing**: 모든 에이전트가 같은 정책 사용
- **PPO 업데이트**: On-policy, stable learning
- **GAE (Generalized Advantage Estimation)**
- **RND 탐험**: 새로운 상태 탐험 보상
- **Reward Shaping**: 생존, 코인 접근, 폭탄 회피 등

**학습 시작**:
```bash
# coin-heaven에서 학습
python main.py play --my-agent mappo_agent --train 1 --no-gui --n-rounds 1000 --scenario coin-heaven

# Self-play (4명 모두 같은 에이전트)
python main.py play --agents mappo_agent mappo_agent mappo_agent mappo_agent --train 1 --no-gui --n-rounds 2000

# 평가
python main.py play --my-agent mappo_agent --n-rounds 5
```

---

## 🎯 중간발표 준비 (4일 계획)

### Day 1 (오늘) ✅
- [x] 베이스라인 측정 스크립트
- [x] MAPPO 설계 문서
- [x] DQN 프로토타입
- [x] MAPPO 프로토타입

### Day 2 (내일)
```bash
# 1. 베이스라인 성능 측정 (1시간)
python benchmark.py

# 2. DQN 학습 시작 (백그라운드)
nohup python main.py play --my-agent dqn_agent --train 1 --no-gui --n-rounds 5000 --scenario coin-heaven > dqn_train.log 2>&1 &

# 3. MAPPO 학습 시작 (백그라운드)
nohup python main.py play --agents mappo_agent mappo_agent mappo_agent mappo_agent --train 1 --no-gui --n-rounds 3000 --scenario coin-heaven > mappo_train.log 2>&1 &

# 4. 학습 진행 모니터링
tail -f dqn_train.log
tail -f mappo_train.log
```

### Day 3
- 학습 결과 수집
- 성능 그래프 생성 (학습 곡선)
- 문제점 파악 및 하이퍼파라미터 튜닝
- 발표 자료 초안 작성

### Day 4
- 발표 자료 완성
- 시연 준비 (GUI 모드 대결)
- 리허설

---

## 📊 발표 자료 구성 (추천)

### 1. 문제 정의 (5분)
- 봄버맨 게임 룰
- 강화학습 문제로서의 특성
- Multi-agent 환경의 도전과제

### 2. 베이스라인 분석 (5분)
- rule_based_agent 성능 측정 결과
- 각 에이전트 비교 (테이블)
- 목표: rule_based를 넘어서기

### 3. 접근 방법론 (10분)
#### DQN 접근
- 7-channel CNN state
- Experience replay
- Rule-guided exploration

#### MAPPO 접근 (핵심!)
- CTDE 아키텍처 설명
- Centralized critic의 이점
- PPO 알고리즘
- Self-play 전략

### 4. 현재 진행상황 (5분)
- 구현 완료 항목
- 초기 학습 결과 (그래프)
- coin-heaven에서의 성능

### 5. 향후 계획 (5분)
- Week 9-10: 기본 학습 완성
- Week 11-12: Curriculum learning
- Week 13-14: Self-play league
- Week 15-16: 최종 튜닝

---

## 🚀 빠른 테스트

```bash
# 1. 베이스라인 측정 (10분)
python benchmark.py --quick

# 2. DQN 에이전트 테스트 (3분)
python main.py play --my-agent dqn_agent --train 1 --no-gui --n-rounds 10 --scenario coin-heaven

# 3. MAPPO 에이전트 테스트 (5분)
python main.py play --my-agent mappo_agent --train 1 --no-gui --n-rounds 10 --scenario coin-heaven

# 4. 대결 시연 (GUI)
python main.py play --agents rule_based_agent dqn_agent mappo_agent random_agent --n-rounds 3
```

---

## 🔧 문제 해결

### PyTorch 설치
```bash
pip install torch torchvision
```

### 학습이 수렴하지 않으면
1. Learning rate 낮추기 (`config.py`)
2. 더 많은 에피소드 수집 (`EPISODES_PER_UPDATE` 증가)
3. Reward scale 조정
4. 간단한 시나리오부터 시작 (`coin-heaven`)

### 메모리 부족
1. Batch size 줄이기
2. Replay buffer 크기 줄이기
3. `--n-rounds` 줄이기

---

## 📈 성공 지표

### 중간발표 기준
- [ ] DQN이 random보다 나음 (coin-heaven)
- [ ] MAPPO가 학습 곡선 보임
- [ ] 자폭률 < 50%
- [ ] 베이스라인 성능 측정 완료

### 최종 대회 기준
- [ ] rule_based 60% 이상 승률
- [ ] 평균 점수 > 15 (classic 4-player)
- [ ] 자폭률 < 10%
- [ ] 전략적 폭탄 설치

---

## 💡 팁

1. **처음엔 coin-heaven으로**: 가장 쉬운 시나리오로 학습 파이프라인 검증
2. **로그 확인**: `agent_code/*/logs/` 에서 학습 진행상황 모니터링
3. **체크포인트 저장**: 주기적으로 저장되므로 중단해도 재개 가능
4. **Tensorboard 사용** (선택): 학습 곡선 시각화
5. **발표에선 솔직하게**: 초기 결과가 완벽하지 않아도 괜찮음. "개선 중"이 중요!

---

**모든 준비 완료! 🎉**

다음 단계: `python benchmark.py`로 베이스라인 측정 시작!
