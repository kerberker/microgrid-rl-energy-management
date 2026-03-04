
"""
Plot Input Profiles (PV and Load)
Generates high-quality PDF plots for the solar and load profiles used in the simulation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from main import setup_environment

def set_style():
    """Configure matplotlib for thesis-grade publication quality."""
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.5)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
    })

def plot_pv_profile(solar_profile, save_path="pv_profile.pdf"):
    """Plot Solar (PV) Generation Profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hours = np.arange(24)
    
    # Area plot
    ax.fill_between(hours, 0, solar_profile, color='#f39c12', alpha=0.3)
    ax.plot(hours, solar_profile, color='#e67e22', label='Solar Generation')
    
    # Peak annotation
    peak_hour = np.argmax(solar_profile)
    peak_val = solar_profile[peak_hour]
    ax.annotate(f'Peak: {peak_val:.2f} kW', 
                xy=(peak_hour, peak_val), 
                xytext=(peak_hour, peak_val + 1),
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontweight='bold')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Daily Solar (PV) Generation Profile', fontweight='bold', pad=15)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, max(solar_profile) * 1.2)
    ax.set_xticks(np.arange(0, 25, 4))
    
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_load_profile(load_profile, save_path="load_profile.pdf"):
    """Plot Load Consumption Profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hours = np.arange(24)
    
    # Area plot
    ax.fill_between(hours, 0, load_profile, color='#3498db', alpha=0.3)
    ax.plot(hours, load_profile, color='#2980b9', label='Load Consumption')
    
    # Peak annotation
    peak_hour = np.argmax(load_profile)
    peak_val = load_profile[peak_hour]
    ax.annotate(f'Peak: {peak_val:.2f} kW', 
                xy=(peak_hour, peak_val), 
                xytext=(peak_hour, peak_val + 1),
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontweight='bold')

    # Mean annotation
    mean_val = np.mean(load_profile)
    ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f} kW')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Daily Load Consumption Profile', fontweight='bold', pad=15)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, max(load_profile) * 1.3) # Increased headroom
    ax.set_xticks(np.arange(0, 25, 4))
    
    ax.legend(loc='upper left') # Moved to upper left to avoid evening peak overlap
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_combined_profiles(solar_profile, load_profile, save_path="combined_profiles.pdf"):
    """Plot Combined PV and Load Profiles on the same page."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    hours = np.arange(24)
    
    # --- Solar Profile (Top) ---
    ax1.fill_between(hours, 0, solar_profile, color='#f39c12', alpha=0.3)
    ax1.plot(hours, solar_profile, color='#e67e22', label='Solar Generation')
    
    # Peak annotation
    peak_hour = np.argmax(solar_profile)
    peak_val = solar_profile[peak_hour]
    ax1.annotate(f'Peak: {peak_val:.2f} kW', 
                xy=(peak_hour, peak_val), 
                xytext=(peak_hour, peak_val + 1),
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontweight='bold')

    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Solar (PV) Generation Profile', fontweight='bold')
    ax1.set_ylim(0, max(solar_profile) * 1.2)
    ax1.grid(True, axis='y')
    ax1.legend(loc='upper right')
    
    # --- Load Profile (Bottom) ---
    ax2.fill_between(hours, 0, load_profile, color='#3498db', alpha=0.3)
    ax2.plot(hours, load_profile, color='#2980b9', label='Load Consumption')
    
    # Peak annotation
    peak_hour = np.argmax(load_profile)
    peak_val = load_profile[peak_hour]
    ax2.annotate(f'Peak: {peak_val:.2f} kW', 
                xy=(peak_hour, peak_val), 
                xytext=(peak_hour, peak_val + 1),
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontweight='bold')

    # Mean annotation
    mean_val = np.mean(load_profile)
    ax2.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f} kW')

    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Load Consumption Profile', fontweight='bold')
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0, max(load_profile) * 1.3) # Increased headroom
    ax2.set_xticks(np.arange(0, 25, 4))
    
    ax2.legend(loc='upper left')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    print("Generating PV and Load Profile Plots...")
    set_style()
    
    # 1. Get Data
    # Use paths relative to current directory or from main config
    DATA_PATH = "electricityConsumptionAndProductioction.csv"
    FALLBACK_DATA_PATH = "PecanStreet_10_Homes_1Min_Data.csv"
    OUTPUT_DIR = "results"
    
    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Retrieve profiles from main setup to match simulation exactly
    solar, load, prices, config = setup_environment(DATA_PATH, FALLBACK_DATA_PATH)
    
    # 2. Plot PV
    plot_pv_profile(solar, f"{OUTPUT_DIR}/pv_profile.pdf")
    
    # 3. Plot Load
    plot_load_profile(load, f"{OUTPUT_DIR}/load_profile.pdf")
    
    # 4. Plot Combined
    plot_combined_profiles(solar, load, f"{OUTPUT_DIR}/input_profiles_combined.pdf")
    
    print("\nDONE.")

if __name__ == "__main__":
    main()
