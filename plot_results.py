
"""
Microgrid Simulation Results Plotting Script
Generates thesis-grade figures for EMS performance analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# =============================================================================
# 1. DATA DEFINITION (USER INPUT SECTION)
# =============================================================================
# Replace these values with your actual simulation results

# Agent Names
AGENTS = ['LP', 'SAC', 'R-SAC', 'PPO', 'Rule-Based']

# Performance Data (0% Noise / Standard Evaluation)
# Values derived from Optimized Simulation Run (Dec 19, 2025 - 30k steps, 24h Horizon)
DATA = {
    'mean_profit':   [-14.22, -15.16, -14.94, -15.91, -34.12], # LP, SAC, R-SAC, PPO, Rule-Based
    'std_profit':    [0.22,   0.25,   0.23,   0.22,   0.27],
    'optimality_gap':[0.0,    6.60,   5.04,   11.93,  139.97],
    
    # Cost Breakdown (Derived from summary)
    'energy_profit': [-13.82, -13.88, -14.20, -14.17, -13.74], 
    'peak_penalty':  [0.0,    0.0,    0.0,    0.0,    0.0],
    'degradation':   [-0.40,  -1.28,  -0.74,  -1.74,  -20.38], # Total - Energy
    
    # Technical Metrics
    'throughput':    [19.91,  39.32,  31.30,  34.39,  18.59],
    'final_soc':     [0.50,   0.51,   0.50,   0.48,   0.10],
}

# Robustness Data (Slope Chart)
# Comparing Profit at 0% vs 30% Forecast Noise
ROBUSTNESS_DATA = {
    '0% Noise':  [-14.70, -15.09, -14.91, -15.97], # LP, SAC, R-SAC, PPO
    '30% Noise': [-16.98, -17.46, -17.14, -19.55],
    'Agents':    ['LP',   'SAC',  'R-SAC', 'PPO']
}

# Profile Data (24-Hour Time Series)
# Extracted from representative simulation day (2025-03-06)
PROFILES = {
    'solar': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.02, 3.53, 5.53, 7.10, 7.63, 7.76, 7.70, 7.04, 5.68, 4.13, 1.01, 0.09, 0.00, 0.00, 0.00, 0.00, 0.00],
    'load': [5.95, 5.87, 5.76, 5.72, 5.85, 6.11, 6.64, 7.20, 6.95, 6.50, 5.87, 5.52, 5.28, 5.34, 5.38, 5.62, 5.84, 6.85, 7.30, 7.82, 7.72, 7.39, 6.72, 6.22],
    'price': [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.15, 0.15, 0.15]
}

# =============================================================================
# 2. GLOBAL STYLE CONFIGURATION
# =============================================================================
def set_style():
    """Configure matplotlib for thesis-grade publication quality."""
    # Use seaborn as a base but override with professional settings
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4) # large font for readability
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
    })

# Consistent Color Palette
COLORS = {
    'LP': '#2c3e50',         # Dark Blue/Grey
    'SAC': '#2980b9',        # Blue
    'R-SAC': '#e67e22',      # Orange (Highlight)
    'PPO': '#27ae60',        # Green
    'Rule-Based': '#c0392b', # Red
    'NoOp': '#7f8c8d'        # Grey
}

# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================

def plot_performance_summary():
    """
    Figure 1: Performance Summary (Dual-Axis)
    Bar: Daily Cost (Absolute) | Line: Optimality Gap
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(AGENTS))
    width = 0.6
    
    # Primary Axis: Daily Cost (Bar)
    # We use inclusive cost (so negative profit becomes positive cost for visualization if easier, 
    # but let's stick to Profit as defined in data, or Cost as requested).
    # Request says "Daily Operational Cost ($) (Absolute values)"
    costs = [-p for p in DATA['mean_profit']] # Convert profit to positive cost
    errors = DATA['std_profit']
    
    bars = ax1.bar(x, costs, width, yerr=errors, capsize=5, 
                   color=[COLORS.get(a, '#333') for a in AGENTS], 
                   alpha=0.9, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Daily Operational Cost ($)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Agent', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(AGENTS)
    ax1.set_ylim(0, max(costs) * 1.2)
    
    # Secondary Axis: Optimality Gap (Line)
    ax2 = ax1.twinx()
    gaps = DATA['optimality_gap']
    ax2.plot(x, gaps, color='#e74c3c', marker='o', linestyle='-', linewidth=3, markersize=8, label='Optimality Gap')
    
    ax2.set_ylabel('Optimality Gap (%)', fontsize=14, fontweight='bold', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(-5, max(gaps) * 1.1)
    
    # Annotations
    for i, v in enumerate(gaps):
        ax2.text(i, v + 2, f"{v:.1f}%", color='#e74c3c', ha='center', fontweight='bold')
        
    plt.title('Economic Performance Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('fig1_performance.pdf', bbox_inches='tight')
    print("Saved fig1_performance.pdf")
    plt.close()


def plot_cost_breakdown():
    """
    Figure 2: Cost Component Breakdown (Stacked Bar)
    Energy vs Peak vs Degradation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(AGENTS))
    width = 0.6
    
    # Components (Positive Cost)
    # Assuming the input data 'energy_profit' is negative profit -> positive cost
    # If the user provides profit, we flip signs.
    
    energy = [-x for x in DATA['energy_profit']]
    peak = [-x for x in DATA['peak_penalty']]
    deg = [-x for x in DATA['degradation']]
    
    # Stacked bars
    p1 = ax.bar(x, energy, width, label='Net Energy Cost', color='#3498db', edgecolor='black', alpha=0.9)
    p2 = ax.bar(x, peak, width, bottom=energy, label='Peak Penalty', color='#e74c3c', edgecolor='black', alpha=0.9)
    # Bottom for p3 is sum of previous two
    bottom_deg = [e + p for e, p in zip(energy, peak)]
    p3 = ax.bar(x, deg, width, bottom=bottom_deg, label='Degradation Cost', color='#f1c40f', edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Daily Cost breakdown ($)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(AGENTS)
    ax.legend(loc='upper left', frameon=True)
    
    plt.title('Cost Component Analysis', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('fig2_breakdown.pdf', bbox_inches='tight')
    print("Saved fig2_breakdown.pdf")
    plt.close()


def plot_efficiency_frontier():
    """
    Figure 3: Efficiency Frontier (Scatter)
    Profit vs Throughput (Risk/Reward)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    profits = DATA['mean_profit']
    throughput = DATA['throughput']
    
    # Plot points
    for i, agent in enumerate(AGENTS):
        color = COLORS.get(agent, 'black')
        ax.scatter(throughput[i], profits[i], s=200, color=color, edgecolor='black', alpha=0.9, zorder=3)
        
        # Add label
        offset_y = 0.5 if i % 2 == 0 else -0.5
        ax.text(throughput[i] + 0.5, profits[i] + offset_y, agent, fontsize=12, fontweight='bold', color=color)
        
    ax.set_xlabel('Battery Throughput (kWh/day)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Daily Profit ($)', fontsize=14, fontweight='bold')
    
    # Draw "Efficient Frontier" arrow or background?
    # Ideally, Top-Left is better (High Profit, Low Throughput/Wear)
    # Or Top-Right if throughput implies activity? 
    # Usually "Less Wear for Same Profit" is better. So Top-Left is Efficiency.
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.title('Efficiency Frontier: Profit vs. Battery Wear', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('fig3_efficiency.pdf', bbox_inches='tight')
    print("Saved fig3_efficiency.pdf")
    plt.close()


def plot_robustness_slope():
    """
    Figure 4: Robustness Analysis (Slope Chart)
    Change in performance from 0% to 30% Noise.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    agents = ROBUSTNESS_DATA['Agents']
    p0 = ROBUSTNESS_DATA['0% Noise']
    p30 = ROBUSTNESS_DATA['30% Noise']
    
    # X-coordinates
    x_pos = [0, 1]
    
    for i, agent in enumerate(agents):
        y_values = [p0[i], p30[i]]
        color = COLORS.get(agent, 'grey')
        
        # Highlight R-SAC
        is_highlight = (agent == 'R-SAC')
        lw = 4 if is_highlight else 2
        alpha = 1.0 if is_highlight else 0.6
        
        # Plot Line
        ax.plot(x_pos, y_values, color=color, linewidth=lw, alpha=alpha, marker='o', markersize=8)
        
        # Annotate labels
        ax.text(-0.05, y_values[0], f"{agent}: ${y_values[0]:.2f}", ha='right', va='center', color=color, fontweight='bold')
        ax.text(1.05, y_values[1], f"${y_values[1]:.2f}", ha='left', va='center', color=color, fontweight='bold')
        
        # Calculate degradation text
        deg_pct = (y_values[1] - y_values[0]) / abs(y_values[0]) * 100
        mid_y = (y_values[0] + y_values[1]) / 2
        
        if is_highlight:
           ax.text(0.5, mid_y + 0.5, f"{agent}\n{deg_pct:.1f}% Drop", ha='center', color=color, fontweight='bold', fontsize=10, backgroundcolor='white')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Ideal (0% Noise)', 'Stochastic (30% Noise)'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily Profit ($)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    
    # Minimalist vertical spines only? No, slope charts usually just lines.
    # Hide y-axis spine?
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True)
    
    plt.title('Robustness Analysis: Performance Under Uncertainty', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('fig4_robustness.pdf', bbox_inches='tight')
    print("Saved fig4_robustness.pdf")
    plt.close()


def plot_system_profiles():
    """
    Figure 5: System Profiles (Dual-Axis)
    Left: Solar/Load Power (kW) | Right: Electricity Price ($/kWh)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    hours = np.arange(24)
    
    # Plot Power (Area/Line)
    ax1.plot(hours, PROFILES['load'], color='#2c3e50', linewidth=2, label='Load Demand', linestyle='--')
    ax1.fill_between(hours, PROFILES['load'], color='#2c3e50', alpha=0.1)
    
    ax1.plot(hours, PROFILES['solar'], color='#f39c12', linewidth=2, label='Solar Generation')
    ax1.fill_between(hours, PROFILES['solar'], color='#f39c12', alpha=0.2)
    
    ax1.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power (kW)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_xlim(0, 23)
    ax1.set_xticks(hours)
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    
    # Secondary Axis for Price
    ax2 = ax1.twinx()
    # Use step plot for ToU prices
    ax2.step(hours, PROFILES['price'], where='post', color='#27ae60', linewidth=2.5, label='ToU Price', zorder=0)
    
    ax2.set_ylabel('Price ($/kWh)', fontsize=14, fontweight='bold', color='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2.set_ylim(0, max(PROFILES['price']) * 1.5)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Combine legends
    # Place legend in upper center or best location
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=3, frameon=False, fontsize=12)
    
    plt.title('Daily Microgrid Profiles & Pricing', fontsize=16, y=1.15)
    plt.tight_layout()
    plt.savefig('fig5_profiles.pdf', bbox_inches='tight')
    print("Saved fig5_profiles.pdf")
    plt.close()


def plot_robustness_scenarios():
    """
    Figure 6: Robustness Evaluation Scenarios
    Visualizes the forecast vs. realized scenarios (uncertainty envelope).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    hours = np.arange(24)
    n_scenarios = 20
    noise_level = 0.30  # 30% noise as used in robust test
    
    # --- Solar Scenarios ---
    base_solar = np.array(PROFILES['solar'])
    ax1.plot(hours, base_solar, color='black', linewidth=3, label='Forecast (Baseline)', zorder=5)
    
    # Generate and plot noisy scenarios
    np.random.seed(42)
    for _ in range(n_scenarios):
        # Noise logic matching MicrogridEnv
        noise = np.random.normal(0, noise_level, 24)
        noisy_solar = base_solar * (1 + noise)
        noisy_solar = np.clip(noisy_solar, 0, None)
        ax1.plot(hours, noisy_solar, color='#f39c12', linewidth=1, alpha=0.3)
        
    ax1.set_title('Solar Uncertainty (30% Noise)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour', fontsize=12)
    ax1.set_ylabel('Power (kW)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Load Scenarios ---
    base_load = np.array(PROFILES['load'])
    ax2.plot(hours, base_load, color='black', linewidth=3, label='Forecast (Baseline)', zorder=5)
    
    for _ in range(n_scenarios):
        noise = np.random.normal(0, noise_level, 24)
        noisy_load = base_load * (1 + noise)
        # Load keeps minimum base
        noisy_load = np.maximum(noisy_load, base_load * 0.5) 
        ax2.plot(hours, noisy_load, color='#2980b9', linewidth=1, alpha=0.3)
        
    ax2.set_title('Load Uncertainty (30% Noise)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel('Power (kW)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Robustness Evaluation Scenarios', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('fig6_robustness_scenarios.pdf', bbox_inches='tight')
    print("Saved fig6_robustness_scenarios.pdf")
    plt.close()


def main():
    print("Generating Thesis-Grade Figures...")
    set_style()
    
    plot_performance_summary()
    plot_cost_breakdown()
    plot_efficiency_frontier()
    plot_robustness_slope()
    plot_system_profiles()
    plot_robustness_scenarios()
    
    print("\nDONE. All figures saved to current directory.")

if __name__ == "__main__":
    main()
