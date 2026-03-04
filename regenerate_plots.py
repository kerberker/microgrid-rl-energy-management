
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from microgrid_env import MicrogridEnv, MicrogridConfig
from stable_baselines3 import SAC, PPO
from rl_agents import RuleBasedAgent, NoopAgent, create_sac_agent, create_ppo_agent
from data_loader import get_tou_prices
from evaluation import run_comprehensive_evaluation
import visualization
import os

def regenerate_plots():
    print("Regenerating plots with enhanced visualization...")
    
    # 1. Setup Data using main.py logic for consistency
    from main import setup_environment
    DATA_PATH = "electricityConsumptionAndProductioction.csv"
    FALLBACK_DATA_PATH = "PecanStreet_10_Homes_1Min_Data.csv"
    
    solar_profile, load_profile, prices, config = setup_environment(DATA_PATH, FALLBACK_DATA_PATH)
    
    # 2. Load Agents
    agents = {}
    
    # R-SAC
    if os.path.exists("results/rsac_model.zip"):
        print("Loading R-SAC...")
        # Create dummy env for loading
        env = MicrogridEnv(solar_profile, load_profile, prices, config=config)
        agents['R-SAC'] = SAC.load("results/rsac_model", env=env)
        
    # SAC
    if os.path.exists("results/sac_model.zip"):
        print("Loading SAC...")
        env = MicrogridEnv(solar_profile, load_profile, prices, config=config)
        agents['SAC'] = SAC.load("results/sac_model", env=env)
        
    # PPO
    if os.path.exists("results/ppo_model.zip"):
        print("Loading PPO...")
        env = MicrogridEnv(solar_profile, load_profile, prices, config=config)
        agents['PPO'] = PPO.load("results/ppo_model", env=env)
    
    # Baselines
    agents['Rule-Based'] = RuleBasedAgent()
    agents['NoOp'] = NoopAgent()
    
    # 3. Run Evaluation (1 Episode for Clarity)
    display_episodes = 50
    print(f"Running evaluation for visualization ({display_episodes} episodes)...")
    results = run_comprehensive_evaluation(
        agents, solar_profile, load_profile, prices, 
        n_episodes=display_episodes, 
        config=config, 
        verbose=True
    )
    
    # 4. Generate Plots
    print("Generating new plots...")
    visualization.generate_all_plots(results, output_dir="results", episode_idx=0)
    print("Done!")

if __name__ == "__main__":
    regenerate_plots()
