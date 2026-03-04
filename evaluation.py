"""
Evaluation Module for Microgrid Energy Management
Multi-episode evaluation and metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

from microgrid_env import MicrogridEnv, MicrogridConfig
from lp_solver import solve_lp_benchmark, LPConfig
from data_loader import get_tou_prices

import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from evaluation."""
    mean_profit: float
    std_profit: float
    min_profit: float
    max_profit: float
    mean_energy_profit: float
    mean_peak_penalty: float
    mean_throughput: float
    mean_final_soc: float
    mean_peak_violations: float
    gap_vs_lp: float
    gap_std: float


def evaluate_agent_single_episode(
    agent,
    env: MicrogridEnv,
    deterministic: bool = True,
    initial_soc: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluate an agent for a single episode.
    
    Args:
        agent: The RL agent (or baseline)
        env: The microgrid environment
        deterministic: Use deterministic policy
        initial_soc: Optional initial SOC (None = random)
        
    Returns:
        Episode results dictionary
    """
    options = {'initial_soc': initial_soc} if initial_soc else None
    obs, info = env.reset(options=options)
    
    total_reward = 0
    done = False
    
    while not done:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    results = env.get_episode_results()
    results['total_reward'] = total_reward
    results['initial_soc'] = initial_soc
    
    return results


def evaluate_agent_multi_episode(
    agent,
    env: MicrogridEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    initial_socs: Optional[List[float]] = None,
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], EvaluationMetrics]:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: The RL agent (or baseline)
        env: The microgrid environment
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        initial_socs: List of initial SOCs (None = uniform random)
        verbose: Print progress
        
    Returns:
        (list of episode results, aggregated metrics)
    """
    if initial_socs is None:
        initial_socs = np.linspace(0.2, 0.8, n_episodes)
    
    all_results = []
    
    for i, init_soc in enumerate(initial_socs):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Episode {i+1}/{n_episodes}")
        
        results = evaluate_agent_single_episode(
            agent, env, deterministic, init_soc
        )
        all_results.append(results)
    
    # Calculate aggregated metrics (convert cost to profit by negating)
    profits = [-r['total_cost'] for r in all_results]  # Negate cost to get profit
    energy_profits = [-r['energy_cost'] for r in all_results]  # Negate for profit
    peak_penalties = [r['peak_penalty'] for r in all_results]
    throughputs = [r['throughput'] for r in all_results]
    final_socs = [r['final_soc'] for r in all_results]
    peak_violations = [r['peak_violations'] for r in all_results]
    
    metrics = EvaluationMetrics(
        mean_profit=np.mean(profits),
        std_profit=np.std(profits),
        min_profit=np.min(profits),
        max_profit=np.max(profits),
        mean_energy_profit=np.mean(energy_profits),
        mean_peak_penalty=np.mean(peak_penalties),
        mean_throughput=np.mean(throughputs),
        mean_final_soc=np.mean(final_socs),
        mean_peak_violations=np.mean(peak_violations),
        gap_vs_lp=0.0,  # Will be calculated separately
        gap_std=0.0
    )
    
    return all_results, metrics


def calculate_lp_benchmark(
    solar_profile: np.ndarray,
    load_profile: np.ndarray,
    price_profile: np.ndarray,
    initial_socs: List[float]
) -> List[Dict[str, Any]]:
    """
    Calculate LP benchmark for multiple initial SOCs.
    
    Args:
        solar_profile: 24-hour solar generation
        load_profile: 24-hour load consumption
        price_profile: 24-hour prices
        initial_socs: List of initial SOCs
        
    Returns:
        List of LP solutions
    """
    lp_results = []
    
    for init_soc in initial_socs:
        result = solve_lp_benchmark(
            solar_profile, load_profile, price_profile, init_soc
        )
        result['initial_soc'] = init_soc
        lp_results.append(result)
    
    return lp_results


def calculate_optimality_gap(
    agent_profits: List[float],
    lp_profits: List[float]
) -> Tuple[float, float, List[float]]:
    """
    Calculate optimality gap between agent and LP benchmark.
    
    Gap = (LP Profit - Agent Profit) / |LP Profit| * 100%
    Positive gap means agent earns less than LP optimal.
    
    Args:
        agent_profits: List of agent profits per episode
        lp_profits: List of LP profits per episode
        
    Returns:
        (mean_gap, std_gap, list of gaps)
    """
    gaps = []
    for agent_profit, lp_profit in zip(agent_profits, lp_profits):
        if abs(lp_profit) > 0.01:  # Avoid division by near-zero
            gap = (lp_profit - agent_profit) / abs(lp_profit) * 100
        else:
            gap = 0.0
        gaps.append(gap)
    
    return np.mean(gaps), np.std(gaps), gaps


def run_comprehensive_evaluation(
    agents: Dict[str, Any],
    solar_profile: np.ndarray,
    load_profile: np.ndarray,
    price_profile: np.ndarray,
    n_episodes: int = 10,
    config: Optional[MicrogridConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of all agents including LP benchmark.
    
    Args:
        agents: Dictionary of {name: agent}
        solar_profile: 24-hour solar generation
        load_profile: 24-hour load consumption
        price_profile: 24-hour prices
        n_episodes: Number of evaluation episodes
        config: Environment configuration
        verbose: Print progress
        
    Returns:
        Comprehensive evaluation results
    """
    # Create evaluation environment
    env_config = config or MicrogridConfig()
    
    # Generate consistent initial SOCs for fair comparison
    initial_socs = list(np.linspace(0.2, 0.8, n_episodes))
    
    results = {}
    
    # Evaluate LP benchmark first
    if verbose:
        print("\n" + "="*60)
        print("Evaluating LP Benchmark...")
        print("="*60)
    
    start_time = time.time()
    lp_results = calculate_lp_benchmark(
        solar_profile, load_profile, price_profile, initial_socs
    )
    lp_time = time.time() - start_time
    
    lp_costs = [r['total_cost'] for r in lp_results]
    lp_profits = [-c for c in lp_costs]  # Convert cost to profit
    
    results['LP'] = {
        'episodes': lp_results,
        'mean_profit': np.mean(lp_profits),
        'std_profit': np.std(lp_profits),
        'mean_gap': 0.0,  # LP is the benchmark
        'evaluation_time': lp_time
    }
    
    if verbose:
        print(f"  Mean profit: ${np.mean(lp_profits):.2f} ± ${np.std(lp_profits):.2f}")
    
    
    import gymnasium as gym
    class ManualNormalizeWrapper(gym.Wrapper):
        """Wrapper to normalize observations using saved stats."""
        def __init__(self, env, obs_rms):
            super().__init__(env)
            self.obs_rms = obs_rms
            self.epsilon = 1e-8
            
        def normalize(self, obs):
            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -10, 10)
            
        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            return self.normalize(obs), r, term, trunc, info
            
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self.normalize(obs), info
            
        def get_episode_results(self):
            return self.env.get_episode_results()

    # Evaluate each agent
    for agent_name, agent in agents.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {agent_name}...")
            print("="*60)
        
        # Create fresh environment for each agent
        env = MicrogridEnv(
            solar_profile, load_profile, price_profile, config=env_config
        )
        
        # Apply normalization for PPO
        if agent_name == 'PPO':
            stats_path = "results/ppo_vec_normalize.pkl"
            if os.path.exists(stats_path):
                print("  Loading PPO normalization stats...")
                # Load dummy vec env to get stats
                dummy = DummyVecEnv([lambda: MicrogridEnv(solar_profile, load_profile, price_profile, config=env_config)])
                vec_norm = VecNormalize.load(stats_path, dummy)
                env = ManualNormalizeWrapper(env, vec_norm.obs_rms)
            else:
                print("  WARNING: PPO normalization stats not found! Agent may perform poorly.")
        
        start_time = time.time()
        agent_results, metrics = evaluate_agent_multi_episode(
            agent, env, n_episodes,
            deterministic=True,
            initial_socs=initial_socs,
            verbose=verbose
        )
        eval_time = time.time() - start_time
        
        # Calculate gap vs LP
        agent_profits = [-r['total_cost'] for r in agent_results]  # Convert cost to profit
        mean_gap, std_gap, gaps = calculate_optimality_gap(agent_profits, lp_profits)
        
        results[agent_name] = {
            'episodes': agent_results,
            'metrics': metrics,
            'mean_profit': metrics.mean_profit,
            'std_profit': metrics.std_profit,
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'gaps': gaps,
            'evaluation_time': eval_time
        }
        
        if verbose:
            print(f"  Mean profit: ${metrics.mean_profit:.2f} ± ${metrics.std_profit:.2f}")
            print(f"  Gap vs LP: {mean_gap:.1f}% ± {std_gap:.1f}%")
    
    return {
        'all_results': results,
        'solar_profile': solar_profile,
        'load_profile': load_profile,
        'price_profile': price_profile,
        'initial_socs': initial_socs,
        'n_episodes': n_episodes
    }


def create_summary_table(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table of evaluation results.
    
    Args:
        evaluation_results: Results from run_comprehensive_evaluation
        
    Returns:
        DataFrame with summary statistics
    """
    all_results = evaluation_results['all_results']
    
    rows = []
    for name, results in all_results.items():
        if name == 'LP':
            row = {
                'Agent': name,
                'Mean Profit ($)': results['mean_profit'],
                'Std Profit ($)': results['std_profit'],
                'Gap vs LP (%)': 0.0,
                'Gap Std (%)': 0.0,
                'Mean Energy Profit ($)': -np.mean([e['energy_cost'] for e in results['episodes']]),
                'Mean Peak Penalty ($)': np.mean([e['peak_penalty'] for e in results['episodes']]),
                'Mean Throughput (kWh)': np.mean([e['throughput'] for e in results['episodes']]),
                'Mean Final SOC': np.mean([e['final_soc'] for e in results['episodes']]),
                'Eval Time (s)': results['evaluation_time']
            }
        else:
            metrics = results['metrics']
            row = {
                'Agent': name,
                'Mean Profit ($)': results['mean_profit'],
                'Std Profit ($)': results['std_profit'],
                'Gap vs LP (%)': results['mean_gap'],
                'Gap Std (%)': results.get('std_gap', 0),
                'Mean Energy Profit ($)': metrics.mean_energy_profit,
                'Mean Peak Penalty ($)': metrics.mean_peak_penalty,
                'Mean Throughput (kWh)': metrics.mean_throughput,
                'Mean Final SOC': metrics.mean_final_soc,
                'Eval Time (s)': results['evaluation_time']
            }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Mean Profit ($)', ascending=False)  # Higher profit = better
    return df


def save_evaluation_results(
    evaluation_results: Dict[str, Any],
    output_path: str
):
    """
    Save evaluation results to CSV files.
    
    Creates two files:
    1. output_path (e.g., results/evaluation_results.csv) - Summary table
    2. output_path_detailed (e.g., results/evaluation_episodes.csv) - Per-episode details
    """
    # 1. Save Summary
    summary = create_summary_table(evaluation_results)
    summary.to_csv(output_path, index=False)
    print(f"Summary results saved to {output_path}")
    
    # 2. Save Detailed Episode Results
    all_results = evaluation_results['all_results']
    detailed_rows = []
    
    for agent_name, agent_data in all_results.items():
        episodes = agent_data['episodes']
        
        for i, episode in enumerate(episodes):
            row = {
                'Agent': agent_name,
                'Episode': i + 1,
                'Initial SOC': episode.get('initial_soc', np.nan),
                'Final SOC': episode.get('final_soc', np.nan),
                'Total Cost ($)': episode['total_cost'],
                'Profit ($)': -episode['total_cost'], # Profit is negative cost
                'Energy Cost ($)': episode['energy_cost'],
                'Peak Penalty ($)': episode['peak_penalty'],
                'Degradation Cost ($)': episode['degradation_cost'],
                'Throughput (kWh)': episode['throughput'],
                'Peak Violations': episode['peak_violations'],
                'Total Solar (kWh)': np.sum(episode.get('actual_solar_profile', episode.get('solar_profile', []))),
                'Total Load (kWh)': np.sum(episode.get('actual_load_profile', episode.get('load_profile', []))),
                'Net Grid Energy (kWh)': np.sum(episode['grid_power']), # Positive = Buy, Negative = Sell
            }
            detailed_rows.append(row)
            
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Construct detailed filename
    base_path = Path(output_path)
    detailed_path = base_path.parent / "evaluation_episodes.csv"
    
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed episode results saved to {detailed_path}")


if __name__ == "__main__":
    # Test evaluation
    from rl_agents import RuleBasedAgent, NoopAgent, RandomAgent
    
    np.random.seed(42)
    
    # Create test profiles
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)
    prices = get_tou_prices()
    
    # Create baseline agents
    agents = {
        'NoOp': NoopAgent(),
        'Rule-Based': RuleBasedAgent()
    }
    
    # Run evaluation
    print("Running quick evaluation test...")
    results = run_comprehensive_evaluation(
        agents, solar, load, prices,
        n_episodes=3, verbose=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    summary = create_summary_table(results)
    print(summary.to_string(index=False))
