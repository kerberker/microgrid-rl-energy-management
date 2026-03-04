"""
Robustness Testing Module for Microgrid Energy Management
Compares LP optimizer vs RL agents under forecast uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from microgrid_env import MicrogridEnv, MicrogridConfig, ScenarioGenerator
from lp_solver import solve_lp_benchmark, LPConfig


@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""
    noise_levels: List[float] = None  # Noise levels to test (as fractions)
    n_scenarios: int = 10  # Number of scenarios per noise level
    seed: int = 42
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]





def evaluate_lp_with_forecast_error(
    forecast_solar: np.ndarray,
    forecast_load: np.ndarray,
    actual_solar: np.ndarray,
    actual_load: np.ndarray,
    price_profile: np.ndarray,
    initial_soc: float = 0.5,
    config: Optional[MicrogridConfig] = None
) -> Dict[str, Any]:
    """
    Evaluate LP performance when forecast differs from reality.
    
    LP optimizes based on forecast, but we evaluate on actual data.
    
    Args:
        forecast_solar: Solar forecast (what LP sees)
        forecast_load: Load forecast (what LP sees)
        actual_solar: Actual solar (reality)
        actual_load: Actual load (reality)
        price_profile: Price profile (assumed known)
        initial_soc: Initial battery SOC
        config: Environment config
        
    Returns:
        Results dictionary with costs and trajectory
    """
    # LP plans based on forecast
    lp_plan = solve_lp_benchmark(
        forecast_solar, forecast_load, price_profile, initial_soc
    )
    
    # Execute LP plan on actual environment
    env_config = config or MicrogridConfig()
    env = MicrogridEnv(actual_solar, actual_load, price_profile, config=env_config)
    
    obs, _ = env.reset(options={'initial_soc': initial_soc})
    
    # Execute LP's planned battery actions
    total_reward = 0
    for t in range(24):
        # Get LP's planned battery power and convert to action
        planned_power = lp_plan['battery_power'][t] if t < len(lp_plan['battery_power']) else 0
        action = np.array([planned_power / env_config.p_bat_max])
        action = np.clip(action, -1, 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    results = env.get_episode_results()
    results['planned_vs_actual'] = {
        'forecast_solar': forecast_solar,
        'actual_solar': actual_solar,
        'forecast_load': forecast_load,
        'actual_load': actual_load
    }
    
    return results


def evaluate_agent_with_noise(
    agent,
    forecast_solar: np.ndarray,
    forecast_load: np.ndarray,
    actual_solar: np.ndarray,
    actual_load: np.ndarray,
    price_profile: np.ndarray,
    initial_soc: float = 0.5,
    config: Optional[MicrogridConfig] = None
) -> Dict[str, Any]:
    """
    Evaluate an RL agent under forecast uncertainty.
    
    The agent observes the 'forecast' (base profile) but the environment
    simulates the 'actual' (noisy profile).
    
    Args:
        agent: The RL agent
        forecast_*: Forecast profiles (Observation)
        actual_*: Actual profiles (Physics)
        price_profile: Price profile
        initial_soc: Initial SOC
        config: Environment config
        
    Returns:
        Results dictionary
    """
    env_config = config or MicrogridConfig()
    
    # Initialize environment with explicit Forecast and Actual profiles
    env = MicrogridEnv(
        solar_profile=forecast_solar,
        load_profile=forecast_load,
        price_profile=price_profile,
        actual_solar_profile=actual_solar,
        actual_load_profile=actual_load,
        actual_price_profile=price_profile, # Assuming price is known/fixed for now
        config=env_config
    )
    
    obs, _ = env.reset(options={'initial_soc': initial_soc})
    
    total_reward = 0
    for t in range(24):
        # Agent sees the observation derived from FORECAST (stored in env.solar_profile)
        # No additional gaussian noise is needed here; the "robustness" challenge 
        # is that the State transition (SOC change) will not match the Agent's 
        # internal expectation perfectly because the Physics uses ACTUAL.
        
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return env.get_episode_results()


def run_robustness_test(
    agents: Dict[str, Any],
    solar_profile: np.ndarray,
    load_profile: np.ndarray,
    price_profile: np.ndarray,
    config: Optional[RobustnessConfig] = None,
    env_config: Optional[MicrogridConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive robustness test comparing agents under uncertainty.
    
    Args:
        agents: Dictionary of {name: agent} (must include 'LP' key with value None)
        solar_profile: Base solar generation
        load_profile: Base load consumption
        price_profile: Price profile
        config: Robustness test configuration
        env_config: Environment configuration
        verbose: Print progress
        
    Returns:
        Dictionary with robustness test results
    """
    config = config or RobustnessConfig()
    generator = ScenarioGenerator(seed=config.seed)
    
    results = {
        'noise_levels': config.noise_levels,
        'agents': {},
        'raw_results': {}
    }
    
    # Initialize results structure
    for agent_name in agents.keys():
        results['agents'][agent_name] = {
            'mean_profits': [],
            'std_profits': [],
            'all_profits': []
        }
        results['raw_results'][agent_name] = {}
    
    if verbose:
        print("\n" + "="*60)
        print("ROBUSTNESS TEST: LP vs RL Agents Under Uncertainty")
        print("="*60)
    
    for noise_level in config.noise_levels:
        if verbose:
            print(f"\n--- Noise Level: {noise_level*100:.0f}% ---")
        
        # Generate scenarios
        scenarios = generator.generate_scenarios(
            solar_profile, load_profile, price_profile,
            noise_level, config.n_scenarios
        )
        
        for agent_name, agent in agents.items():
            profits = []
            
            for scenario in scenarios:
                initial_soc = np.random.uniform(0.2, 0.8)
                
                if agent_name == 'LP':
                    # LP plans with base forecast, executes on noisy reality
                    result = evaluate_lp_with_forecast_error(
                        forecast_solar=solar_profile,
                        forecast_load=load_profile,
                        actual_solar=scenario['solar'],
                        actual_load=scenario['load'],
                        price_profile=price_profile,
                        initial_soc=initial_soc,
                        config=env_config
                    )
                else:
                    # RL agents:
                    # Forecast = Base Profile (solar_profile)
                    # Actual = Scenario Profile (scenario['solar'])
                    result = evaluate_agent_with_noise(
                        agent,
                        forecast_solar=solar_profile,
                        forecast_load=load_profile,
                        actual_solar=scenario['solar'],
                        actual_load=scenario['load'],
                        price_profile=price_profile,
                        initial_soc=initial_soc,
                        config=env_config
                    )
                
                profit = -result['total_cost']  # Convert cost to profit
                profits.append(profit)
            
            mean_profit = np.mean(profits)
            std_profit = np.std(profits)
            
            results['agents'][agent_name]['mean_profits'].append(mean_profit)
            results['agents'][agent_name]['std_profits'].append(std_profit)
            results['agents'][agent_name]['all_profits'].append(profits)
            results['raw_results'][agent_name][noise_level] = profits
            
            if verbose:
                print(f"  {agent_name:12s}: ${mean_profit:.2f} ± ${std_profit:.2f}")
    
    return results


def plot_robustness_comparison(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot robustness comparison showing profit vs noise level.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    noise_levels = np.array(results['noise_levels']) * 100  # Convert to percentage
    
    colors = {
        'LP': '#2ecc71',
        'SAC': '#3498db',
        'R-SAC': '#9b59b6',
        'PPO': '#e74c3c',
        'NoOp': '#95a5a6',
        'Rule-Based': '#f39c12'
    }
    
    # Plot 1: Mean profit vs noise level
    for agent_name, agent_results in results['agents'].items():
        color = colors.get(agent_name, '#333333')
        mean_profits = agent_results['mean_profits']
        std_profits = agent_results['std_profits']
        
        ax1.plot(noise_levels, mean_profits, 'o-', label=agent_name, 
                color=color, linewidth=2, markersize=8)
        ax1.fill_between(noise_levels, 
                        np.array(mean_profits) - np.array(std_profits),
                        np.array(mean_profits) + np.array(std_profits),
                        alpha=0.2, color=color)
    
    ax1.set_xlabel('Forecast Error (%)', fontsize=12)
    ax1.set_ylabel('Mean Profit ($)', fontsize=12)
    ax1.set_title('Profit vs Forecast Uncertainty', fontweight='bold', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 2: Relative performance (% of 0% noise performance)
    for agent_name, agent_results in results['agents'].items():
        color = colors.get(agent_name, '#333333')
        mean_profits = np.array(agent_results['mean_profits'])
        
        # Normalize to 0% noise performance
        if abs(mean_profits[0]) > 0.01:
            relative_perf = mean_profits / mean_profits[0] * 100
        else:
            relative_perf = np.ones_like(mean_profits) * 100
        
        ax2.plot(noise_levels, relative_perf, 'o-', label=agent_name,
                color=color, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Forecast Error (%)', fontsize=12)
    ax2.set_ylabel('Relative Performance (%)', fontsize=12)
    ax2.set_title('Performance Degradation Under Uncertainty', fontweight='bold', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.3, label='Baseline')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_robustness_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table of robustness results."""
    rows = []
    
    for agent_name, agent_results in results['agents'].items():
        # Performance at 0% noise (baseline)
        baseline = agent_results['mean_profits'][0]
        
        # Performance at highest noise level
        worst = agent_results['mean_profits'][-1]
        
        # Degradation percentage
        if abs(baseline) > 0.01:
            degradation = (baseline - worst) / abs(baseline) * 100
        else:
            degradation = 0.0
        
        rows.append({
            'Agent': agent_name,
            'Profit @ 0% Noise': baseline,
            'Profit @ 30% Noise': worst,
            'Degradation (%)': degradation,
            'Robustness Score': 100 - abs(degradation)
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Robustness Score', ascending=False)
    return df


def save_robustness_results(
    results: Dict[str, Any],
    output_path: str
):
    """
    Save robustness results to CSV files.
    
    Creates:
    1. output_path (e.g., results/robustness_summary.csv)
    2. detailed_path (e.g., results/robustness_detailed.csv)
    """
    # 1. Save Summary
    summary = create_robustness_summary(results)
    summary.to_csv(output_path, index=False)
    print(f"Robustness summary saved to {output_path}")
    
    # 2. Save Detailed Results (per scenario)
    detailed_rows = []
    
    noise_levels = results['noise_levels']
    raw_results = results['raw_results']
    
    for agent_name, agent_data in raw_results.items():
        for noise_level, profits in agent_data.items():
            for i, profit in enumerate(profits):
                detailed_rows.append({
                    'Agent': agent_name,
                    'Noise Level': noise_level,
                    'Scenario': i + 1,
                    'Profit ($)': profit
                })
                
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Construct detailed filename
    base_path = Path(output_path)
    detailed_path = base_path.parent / "robustness_detailed.csv"
    
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed robustness results saved to {detailed_path}")


if __name__ == "__main__":
    # Test the robustness module
    from data_loader import get_tou_prices
    from rl_agents import NoopAgent, RuleBasedAgent
    from stable_baselines3 import SAC, PPO
    import os
    
    print("Testing Robustness Module...")
    
    # Create test profiles
    np.random.seed(42)
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)
    prices = get_tou_prices()
    
    # Create environment for loading agents
    env_config = MicrogridConfig()
    env = MicrogridEnv(solar, load, prices, config=env_config)
    
    # Base agents
    agents = {
        'LP': None,  # LP is handled specially
        'NoOp': NoopAgent(),
        'Rule-Based': RuleBasedAgent()
    }
    
    # specific paths to checked models
    model_dir = Path("results")
    
    # Load trained agents if they exist
    if (model_dir / "sac_model.zip").exists():
        try:
            print("Loading SAC agent...")
            agents['SAC'] = SAC.load(model_dir / "sac_model", env=env)
        except Exception as e:
            print(f"Failed to load SAC agent: {e}")

    if (model_dir / "ppo_model.zip").exists():
        try:
            print("Loading PPO agent...")
            agents['PPO'] = PPO.load(model_dir / "ppo_model", env=env)
        except Exception as e:
            print(f"Failed to load PPO agent: {e}")
            
    if (model_dir / "rsac_model.zip").exists():
        try:
            print("Loading RSAC agent...")
            agents['R-SAC'] = SAC.load(model_dir / "rsac_model", env=env)
        except Exception as e:
            print(f"Failed to load RSAC agent: {e}")
    
    # Run test
    test_config = RobustnessConfig(
        noise_levels=[0.0, 0.1, 0.2, 0.3],
        n_scenarios=5  # Increased scenarios for better stats
    )
    
    results = run_robustness_test(
        agents, solar, load, prices,
        config=test_config,
        verbose=True
    )
    
    # Create summary
    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    summary = create_robustness_summary(results)
    print(summary.to_string(index=False))
    
    # Test plot
    fig = plot_robustness_comparison(results, save_path="test_robustness.pdf")
    print("\nSaved test plot: test_robustness.pdf")
    plt.close(fig)
