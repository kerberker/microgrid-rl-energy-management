
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from microgrid_env import MicrogridEnv, MicrogridConfig
from data_loader import ElectricityDataLoader, get_tou_prices
import os

def visualize_forecast_behavior():
    print("Visualizing Forecast vs Actual Behavior using Real Data...")
    
    # 1. Load Real Data
    data_path = "electricityConsumptionAndProductioction.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return

    loader = ElectricityDataLoader(data_path)
    daily_profiles = loader.create_daily_profiles()
    
    if not daily_profiles:
        print("Error: No daily profiles created.")
        return
        
    # Select a sample day (e.g., index 0)
    sample_day = daily_profiles[0]
    forecast_solar = sample_day['solar_kw']
    forecast_load = sample_day['load_kw']
    prices = get_tou_prices()
    
    print(f"Loaded sample day: {sample_day['date']}")
    
    # 2. Setup Environment
    config = MicrogridConfig()
    noise_level = 0.30 # 30% noise for clear visualization
    correlation = 0.8
    
    # Initialize Environment with Randomization enabled to generate "Actuals"
    # The 'forecast' passed here is the "real data" which represents our best estimate/baseline
    # The 'actual' will be a perturbed version of this.
    env = MicrogridEnv(
        solar_profile=forecast_solar,
        load_profile=forecast_load,
        price_profile=prices,
        config=config,
        randomize_env=True,  # This will generate noisy actuals internal to the env
        noise_level=noise_level,
        correlation=correlation
    )
    
    # 3. Run Episode to capture "Actuals"
    # We don't need a trained agent to visualize the environment dynamics (forecast vs actual)
    # A random agent is sufficient to step through the environment
    obs, info = env.reset(seed=123) # Specific seed for reproducibility
    
    # Capture data
    actual_solar = env.actual_solar_profile
    actual_load = env.actual_load_profile
    
    # 4. Plotting
    print("Generating plot...")
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-paper')
    # fallback if not available
    # plt.style.use('seaborn-paper') 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    hours = np.linspace(0, 23, 24)
    
    # Plot 1: Solar Forecast vs Actual
    ax1.plot(hours, forecast_solar, 'b--', linewidth=2, label='Forecast (Expected)')
    ax1.plot(hours, actual_solar, 'b-', linewidth=2, alpha=0.7, label='Actual (Realized)')
    ax1.fill_between(hours, forecast_solar, actual_solar, alpha=0.2, color='blue', label='Forecast Error')
    ax1.set_ylabel('Solar Generation (kW)', fontsize=12)
    ax1.set_title(f'Solar Generation: Forecast vs Actual (Noise $\sigma$={noise_level:.0%})', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Load Forecast vs Actual
    ax2.plot(hours, forecast_load, 'r--', linewidth=2, label='Forecast (Expected)')
    ax2.plot(hours, actual_load, 'r-', linewidth=2, alpha=0.7, label='Actual (Realized)')
    ax2.fill_between(hours, forecast_load, actual_load, alpha=0.2, color='red', label='Forecast Error')
    ax2.set_ylabel('Load Consumption (kW)', fontsize=12)
    ax2.set_title('Load Consumption: Forecast vs Actual', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(hours)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs("Pages/fig", exist_ok=True)
    
    save_path = "Pages/fig/forecast_vs_actual.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_forecast_behavior()
