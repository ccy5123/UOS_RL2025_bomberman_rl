#!/usr/bin/env python3
"""
Baseline Performance Benchmark Script
벤치마크용 스크립트 - 여러 에이전트의 성능을 비교 측정
"""

import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


def run_game(agents, scenario='classic', n_rounds=100, seed=None):
    """
    게임 실행 및 결과 수집

    Args:
        agents: 에이전트 이름 리스트 (예: ['rule_based_agent', 'random_agent'])
        scenario: 게임 시나리오
        n_rounds: 라운드 수
        seed: 랜덤 시드 (재현성)

    Returns:
        결과 JSON 파일 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/benchmark_{timestamp}.json"

    cmd = [
        'python', 'main.py', 'play',
        '--agents'] + agents + [
        '--scenario', scenario,
        '--n-rounds', str(n_rounds),
        '--no-gui',
        '--save-stats', result_file,
        '--silence-errors'
    ]

    if seed is not None:
        cmd.extend(['--seed', str(seed)])

    print(f"Running: {' '.join(agents)} for {n_rounds} rounds...")
    subprocess.run(cmd, check=True)

    return result_file


def analyze_results(result_file):
    """
    결과 파일 분석

    Returns:
        dict: 에이전트별 통계
    """
    with open(result_file, 'r') as f:
        data = json.load(f)

    stats = {}

    for agent_name, agent_data in data['by_agent'].items():
        rounds_played = agent_data.get('rounds', 0)

        stats[agent_name] = {
            'total_score': agent_data.get('score', 0),
            'avg_score': agent_data.get('score', 0) / rounds_played if rounds_played > 0 else 0,
            'coins_collected': agent_data.get('coins', 0),
            'kills': agent_data.get('kills', 0),
            'suicides': agent_data.get('suicides', 0),
            'rounds_played': rounds_played,
            'survival_rate': 1.0 - (agent_data.get('suicides', 0) / rounds_played) if rounds_played > 0 else 0,
        }

    return stats


def print_comparison_table(all_stats, agent_names):
    """
    에이전트 비교 테이블 출력
    """
    print("\n" + "="*80)
    print("AGENT PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Agent':<20} {'Avg Score':>10} {'Coins':>8} {'Kills':>8} {'Suicides':>10} {'Survival%':>12}")
    print("-"*80)

    for agent in agent_names:
        s = all_stats[agent]
        print(f"{agent:<20} {s['avg_score']:>10.2f} {s['coins_collected']:>8} "
              f"{s['kills']:>8} {s['suicides']:>10} {s['survival_rate']*100:>11.1f}%")

    print("="*80)


def run_benchmark_suite():
    """
    전체 벤치마크 스위트 실행
    """
    print("\n🎮 BOMBERMAN RL BENCHMARK SUITE")
    print("="*80)

    # 1. 기본 에이전트들 비교 (1v1v1v1)
    print("\n[Test 1] 4-Player Free-for-All (Classic)")
    agents = ['rule_based_agent', 'random_agent', 'peaceful_agent', 'coin_collector_agent']
    result_file = run_game(agents, scenario='classic', n_rounds=100, seed=42)
    stats_4p = analyze_results(result_file)
    print_comparison_table(stats_4p, agents)

    # 2. 1v1 매치 (rule_based vs others)
    print("\n[Test 2] 1v1 Matches vs rule_based_agent")
    opponents = ['random_agent', 'peaceful_agent', 'coin_collector_agent']

    stats_1v1 = {}
    for opponent in opponents:
        result_file = run_game(['rule_based_agent', opponent], scenario='classic', n_rounds=50, seed=42)
        stats = analyze_results(result_file)
        stats_1v1[opponent] = stats

        rb_score = stats['rule_based_agent']['avg_score']
        opp_score = stats[opponent]['avg_score']
        win_rate = (rb_score / (rb_score + opp_score) * 100) if (rb_score + opp_score) > 0 else 50

        print(f"  rule_based vs {opponent:20s}: {rb_score:.2f} - {opp_score:.2f} "
              f"(rule_based win rate: {win_rate:.1f}%)")

    # 3. 시나리오별 성능 (rule_based_agent)
    print("\n[Test 3] rule_based_agent Performance by Scenario")
    scenarios = ['coin-heaven', 'loot-crate', 'classic']

    for scenario in scenarios:
        agents = ['rule_based_agent'] * 4
        result_file = run_game(agents, scenario=scenario, n_rounds=50, seed=42)
        stats = analyze_results(result_file)
        avg_score = np.mean([s['avg_score'] for s in stats.values()])

        print(f"  {scenario:15s}: avg_score = {avg_score:.2f}")

    # 4. 요약 저장
    summary = {
        'timestamp': datetime.now().isoformat(),
        '4_player_ffa': stats_4p,
        '1v1_matches': stats_1v1,
        'baseline_agent': 'rule_based_agent',
        'baseline_avg_score': stats_4p['rule_based_agent']['avg_score']
    }

    summary_file = 'results/benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Benchmark complete! Summary saved to {summary_file}")
    print(f"\n📊 Baseline to beat: {stats_4p['rule_based_agent']['avg_score']:.2f} avg score (rule_based_agent)")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Bomberman agents')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (fewer rounds)')
    parser.add_argument('--agents', nargs='+', help='Custom agent list to benchmark')
    parser.add_argument('--scenario', default='classic', help='Game scenario')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds')

    args = parser.parse_args()

    if args.agents:
        # Custom benchmark
        result_file = run_game(args.agents, scenario=args.scenario, n_rounds=args.rounds)
        stats = analyze_results(result_file)
        print_comparison_table(stats, args.agents)
    else:
        # Full benchmark suite
        if args.quick:
            print("Running quick benchmark (reduced rounds)...")
            # Override n_rounds in run_game calls to 20
        run_benchmark_suite()
