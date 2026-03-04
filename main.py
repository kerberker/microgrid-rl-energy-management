"""
Microgrid Energy Management System - Main Pipeline
Complete simulation with RL agents (SAC, PPO, R-SAC), LP benchmark, and visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import custom modules
from data_loader import PecanStreetDataLoader, ElectricityDataLoader, get_tou_prices
from microgrid_env import MicrogridEnv, MicrogridConfig, DynamicMicrogridEnv
from rl_agents import (
    create_sac_agent, create_ppo_agent, create_robust_sac_agent,
    train_agent, save_agent, evaluate_agent,
    NoopAgent, RuleBasedAgent, TrainingCallback
)
from lp_solver import solve_lp_benchmark, LPConfig
from evaluation import run_comprehensive_evaluation, create_summary_table, save_evaluation_results
from visualization import generate_all_plots
from robustness_test import run_robustness_test, plot_robustness_comparison, save_robustness_results, create_robustness_summary, RobustnessConfig

import matplotlib.pyplot as plt


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def setup_environment(data_path: str, fallback_path: str):
    """
    Setup the microgrid environment configuration and load data.
    
    Returns:
        tuple: (solar_profile, load_profile, price_profile, config)
    """
    print_header("Setting Up Environment")
    
    daily_profiles = None
    
    # Try loading the electricity consumption/production data first
    try:
        data_loader = ElectricityDataLoader(data_path)
        data_loader.load_raw_data()
        data_loader.process_data()
        data_loader.create_daily_profiles()
        print(f"\n[OK] Using {data_path}")
        daily_profiles = data_loader.daily_profiles
    except FileNotFoundError:
        print(f"Warning: {data_path} not found. Trying Pecan Street data...")
        try:
            data_loader = PecanStreetDataLoader(fallback_path)
            data_loader.load_raw_data()
            data_loader.process_data()
            daily_profiles = data_loader.create_daily_profiles()
            print(f"\n[OK] Using {fallback_path}")
        except FileNotFoundError:
            print(f"Warning: {fallback_path} not found. Using synthetic data.")
    
    # Get price profile
    prices = get_tou_prices()
    
    # If we have real data, use a representative day
    if daily_profiles and len(daily_profiles) > 0:
        # Use a day with good solar production (e.g., a summer day)
        # Find day with highest solar generation
        best_idx = 0
        best_solar = 0
        for idx, profile in enumerate(daily_profiles):
            total_solar = profile['solar_kw'].sum()
            if total_solar > best_solar:
                best_solar = total_solar
                best_idx = idx
        
        profile = daily_profiles[best_idx]
        solar = profile['solar_kw'] * 5  # Scale up solar for realistic residential system (~7.5 kW peak)
        load = profile['load_kw']
        print(f"\nUsing profile from Date: {profile['date']} (highest solar day)")
        print(f"Solar scaled 5x for realistic residential microgrid")
    else:
        # Generate synthetic data
        print("\nUsing synthetic solar and load profiles")
        np.random.seed(42)
        solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)) + np.random.normal(0, 0.5, 24))
        load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4) + np.random.normal(0, 0.3, 24)
        solar = np.clip(solar, 0, 10)
        load = np.clip(load, 1, 10)
    
    print(f"Solar generation: min={solar.min():.2f}, max={solar.max():.2f}, mean={solar.mean():.2f} kW")
    print(f"Load consumption: min={load.min():.2f}, max={load.max():.2f}, mean={load.mean():.2f} kW")
    
    # Environment Configuration
    config = MicrogridConfig(
        e_max=13.5,          # 13.5 kWh battery (Tesla Powerwall)
        e_min_ratio=0.1,     # 10% minimum SOC
        p_bat_max=5.0,       # 5 kW max charge/discharge
        eta_charge=0.95,     # 95% charging efficiency
        eta_discharge=0.95,  # 95% discharging efficiency
        ramp_rate=2.5,       # 2.5 kW/hour max ramp
        p_grid_peak=10.0,    # 10 kW peak threshold
        peak_penalty_rate=0.50,  # $0.50/kW peak penalty
        degradation_cost_per_kwh=0.02,  # $0.02/kWh degradation
        forecast_horizon=24,   # 24-hour lookahead (Full day visibility)
    )
    
    return solar, load, prices, config


def main():
    """
    Main function to run the complete microgrid simulation.
    """
    print_header("MICROGRID ENERGY MANAGEMENT SYSTEM WITH RL")
    print("Starting simulation pipeline...")
    print(f"Time: {pd.Timestamp.now()}")
    
    # Configuration
    DATA_PATH = "electricityConsumptionAndProductioction.csv"
    FALLBACK_DATA_PATH = "PecanStreet_10_Homes_1Min_Data.csv"
    OUTPUT_DIR = "results"
    
    # Training parameters (Final Run - increased for better results)
    SAC_TIMESTEPS = 200000
    PPO_TIMESTEPS = 200000
    N_EVAL_EPISODES = 100
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1 & 2: Environment Setup
    # =========================================================================
    solar, load, prices, config = setup_environment(DATA_PATH, FALLBACK_DATA_PATH)
    
    # Create training environment
    train_env = MicrogridEnv(solar, load, prices, config=config)
    
    print(f"Environment created with:")
    print(f"  - Observation space: {train_env.observation_space.shape}")
    print(f"  - Action space: {train_env.action_space.shape}")
    print(f"  - Battery capacity: {config.e_max} kWh")
    print(f"  - Max charge/discharge: {config.p_bat_max} kW")
    
    # =========================================================================
    # STEP 3: Train RL Agents
    # =========================================================================
    print_header("STEP 3: Training RL Agents")
    
    trained_agents = {}
    
    # --- Train SAC ---
    # --- Train SAC ---
    print("\n[1/3] Training SAC Agent...")
    sac_env = MicrogridEnv(solar, load, prices, config=config)
    sac_agent = create_sac_agent(sac_env, verbose=0)
    sac_callback = TrainingCallback(log_interval=2000)
    
    start_time = time.time()
    train_agent(sac_agent, SAC_TIMESTEPS, callback=sac_callback, progress_bar=True)
    sac_train_time = time.time() - start_time
    
    save_agent(sac_agent, f"{OUTPUT_DIR}/sac_model")
    trained_agents['SAC'] = sac_agent
    print(f"  SAC training completed in {sac_train_time:.1f}s")
    
    # Load existing SAC (Commented out)
    # from rl_agents import load_agent
    # sac_env_dummy = MicrogridEnv(solar, load, prices, config=config)
    # sac_agent = load_agent(type(create_sac_agent(sac_env_dummy)), f"{OUTPUT_DIR}/sac_model", sac_env_dummy)
    # trained_agents['SAC'] = sac_agent
    # sac_train_time = 0
    # print(f"  SAC training completed in {sac_train_time:.1f}s")
    
    # --- Train Robust SAC ---
    print("\n[2/3] Training Robust SAC Agent...")
    # Enable Domain Randomization for R-SAC training
    # This exposes the agent to many variations of the day, helping it learn robust policies
    rsac_env = MicrogridEnv(
        solar, load, prices,
        config=config, 
        randomize_env=True, 
        noise_level=0.30,  # Train on up to 30% noise scenarios
        variable_noise_level=True, # Enable dynamic noise sampling
        correlation=0.9  # Use high correlation to simulate consistent forecast errors
    )
    rsac_agent, rsac_wrapped_env = create_robust_sac_agent(
        rsac_env, 
        verbose=0
    )
    rsac_callback = TrainingCallback(log_interval=2000)
    
    start_time = time.time()
    train_agent(rsac_agent, SAC_TIMESTEPS, callback=rsac_callback, progress_bar=True)
    rsac_train_time = time.time() - start_time
    
    save_agent(rsac_agent, f"{OUTPUT_DIR}/rsac_model")
    trained_agents['R-SAC'] = rsac_agent
    print(f"  R-SAC training completed in {rsac_train_time:.1f}s")
    
    # --- Train PPO ---
    print("\n[3/3] Training PPO Agent...")
    # PPO requires normalized environment for good performance
    ppo_env = DummyVecEnv([lambda: MicrogridEnv(solar, load, prices, config=config)])
    ppo_env = VecNormalize(ppo_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    ppo_agent = create_ppo_agent(ppo_env, verbose=0)
    ppo_callback = TrainingCallback(log_interval=3000)
    
    start_time = time.time()
    train_agent(ppo_agent, PPO_TIMESTEPS, callback=ppo_callback, progress_bar=True)
    ppo_train_time = time.time() - start_time
    
    save_agent(ppo_agent, f"{OUTPUT_DIR}/ppo_model")
    ppo_env.save(f"{OUTPUT_DIR}/ppo_vec_normalize.pkl") # Save normalization stats
    trained_agents['PPO'] = ppo_agent
    print(f"  PPO training completed in {ppo_train_time:.1f}s")
    
    # Add baseline agents
    trained_agents['NoOp'] = NoopAgent()
    trained_agents['Rule-Based'] = RuleBasedAgent()
    
    print(f"\nTotal training time: {sac_train_time + rsac_train_time + ppo_train_time:.1f}s")
    
    # =========================================================================
    # STEP 4: Comprehensive Evaluation
    # =========================================================================
    print_header("STEP 4: Evaluating All Agents")
    
    evaluation_results = run_comprehensive_evaluation(
        agents=trained_agents,
        solar_profile=solar,
        load_profile=load,
        price_profile=prices,
        n_episodes=N_EVAL_EPISODES,
        config=config,
        verbose=True
    )
    
    # Save evaluation results
    save_evaluation_results(evaluation_results, f"{OUTPUT_DIR}/evaluation_results.csv")
    
    # =========================================================================
    # STEP 5: Generate Visualizations
    # =========================================================================
    print_header("STEP 5: Generating Visualizations")
    
    figures = generate_all_plots(
        evaluation_results,
        output_dir=OUTPUT_DIR,
        episode_idx=0
    )
    
    # =========================================================================
    # STEP 6: Robustness Testing
    # =========================================================================
    print_header("STEP 6: Robustness Testing (LP vs RL Under Uncertainty)")
    
    # Prepare agents for robustness test
    robustness_agents = {
        'LP': None,  # LP is handled specially
        'SAC': trained_agents['SAC'],
        'R-SAC': trained_agents['R-SAC'],
        'PPO': trained_agents['PPO']
    }
    
    robustness_config = RobustnessConfig(
        noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30],
        n_scenarios=10
    )
    
    robustness_results = run_robustness_test(
        agents=robustness_agents,
        solar_profile=solar,
        load_profile=load,
        price_profile=prices,
        config=robustness_config,
        env_config=config,
        verbose=True
    )
    
    # Generate robustness plots
    robustness_fig = plot_robustness_comparison(
        robustness_results,
        save_path=f"{OUTPUT_DIR}/13_robustness_comparison.pdf"
    )
    plt.close(robustness_fig)
    
    # Save robustness results
    save_robustness_results(
        robustness_results,
        output_path=f"{OUTPUT_DIR}/robustness_summary.csv"
    )
    
    # Print robustness summary
    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    robustness_summary = create_robustness_summary(robustness_results)
    print(robustness_summary.to_string(index=False))
    
    # =========================================================================
    # STEP 7: Summary
    # =========================================================================
    print_header("SIMULATION COMPLETE - SUMMARY")
    
    summary_df = create_summary_table(evaluation_results)
    print(summary_df.to_string(index=False))
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"   - 13 visualization plots (PNG)")
    print(f"   - evaluation_results.csv")
    print(f"   - Trained models (SAC, R-SAC, PPO)")
    
    # Key findings
    print("\nKey Findings:")
    all_results = evaluation_results['all_results']
    
    lp_profit = all_results['LP']['mean_profit']
    print(f"   - LP Benchmark profit: ${lp_profit:.2f}")
    
    best_rl = max(
        [(name, res['mean_profit']) for name, res in all_results.items() if name not in ['LP', 'NoOp', 'Rule-Based']],
        key=lambda x: x[1]
    )
    print(f"   - Best RL agent: {best_rl[0]} (${best_rl[1]:.2f})")
    
    if abs(lp_profit) > 0.01:
        gap = (lp_profit - best_rl[1]) / abs(lp_profit) * 100
        print(f"   - Gap vs LP: {gap:.1f}%")
    
    # Robustness finding
    print("\nRobustness Findings:")
    most_robust = robustness_summary.iloc[0]['Agent']
    print(f"   - Most robust agent: {most_robust}")
    
    plt.close('all')  # Close all figures to free memory
    
    return evaluation_results, robustness_results


def run_quick_demo():
    """
    Run a quick demonstration with reduced training steps.
    Useful for testing the pipeline.
    """
    print_header("QUICK DEMO MODE")
    
    np.random.seed(42)
    
    # Synthetic profiles
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)
    prices = get_tou_prices()
    
    config = MicrogridConfig()
    
    # Create environment
    env = MicrogridEnv(solar, load, prices, config=config)
    
    # Quick SAC training
    print("Training SAC (500 steps)...")
    sac_agent = create_sac_agent(env, verbose=0)
    train_agent(sac_agent, 500, progress_bar=False)
    
    # Evaluate
    agents = {
        'SAC': sac_agent,
        'Rule-Based': RuleBasedAgent()
    }
    
    print("\nEvaluating...")
    results = run_comprehensive_evaluation(
        agents, solar, load, prices,
        n_episodes=3, config=config, verbose=False
    )
    
    print("\nSummary:")
    summary = create_summary_table(results)
    print(summary.to_string(index=False))
    
    # Generate one plot
    from visualization import plot_soc_comparison
    fig = plot_soc_comparison(results, save_path="demo_soc.png")
    print("\nSaved demo plot: demo_soc.png")
    plt.close(fig)


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_quick_demo()
    else:
        results = main()
