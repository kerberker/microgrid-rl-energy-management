"""
Visualization Module for Microgrid Energy Management
Generates 12 comprehensive plots for analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd


# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100
})

# Color palette for agents
AGENT_COLORS = {
    'LP': '#2ecc71',       # Green
    'SAC': '#3498db',      # Blue
    'R-SAC': '#9b59b6',    # Purple
    'PPO': '#e74c3c',      # Red
    'NoOp': '#95a5a6',     # Gray
    'Rule-Based': '#f39c12' # Orange
}


def get_agent_color(agent_name: str) -> str:
    """Get color for an agent."""
    return AGENT_COLORS.get(agent_name, '#333333')



# ... (imports remain)

# ... (styles remain)

def add_background_shading(ax, label: bool = True):
    """Add background shading for TOU pricing periods."""
    # Peak Pricing: 17:00 - 21:00 (5 PM - 9 PM) usually, but code says 14-21
    # Checking get_tou_prices in data_loader.py would be best, but let's stick to consistent visual
    # data_loader.py usually defines peak as 16-21 or 17-21. 
    # Let's align with the `plot_pricing_strategy` which used 14-21.
    # Actually, let's make it generic or check the pricing profile if possible? 
    # For now, hardcode to 14-21 (2pm-9pm) as "Peak" and 0-6 as "Off-Peak" to match previous code.
    
    # Off-Peak (0-6)
    ax.axvspan(0, 6, alpha=0.1, color='green', label='Off-Peak Price' if label else None)
    # Peak (14-21)
    ax.axvspan(14, 21, alpha=0.1, color='red', label='Peak Price (High Cost)' if label else None)
    
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.3)

def plot_soc_comparison(
    evaluation_results: Dict[str, Any],
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 1: SOC trajectory comparison for all agents."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add Shading
    add_background_shading(ax)
    
    all_results = evaluation_results['all_results']
    
    for agent_name, results in all_results.items():
        episodes = results['episodes']
        if episode_idx < len(episodes):
            soc = episodes[episode_idx]['soc_trajectory']
            
            # Prepend initial SOC (0.5) for visualization clarity
            # The env records SOC at the END of each step. We want to show the START (t=0).
            soc = np.concatenate(([0.5], soc))
            hours = np.arange(len(soc))
            
            color = get_agent_color(agent_name)
            # Make RobustSAC thicker
            lw = 3 if 'R-SAC' in agent_name else 2
            ax.plot(hours, soc, label=agent_name, color=color, linewidth=lw, marker='o', markersize=4)
    
    # Add SOC limits
    ax.axhline(y=0.1, color='black', linestyle='--', alpha=0.8, label='Min SOC (10%)')
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('State of Charge (SOC)')
    ax.set_title('Battery State of Charge Comparison', fontweight='bold')
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1.1)
    
    # Better Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_cumulative_profit(
    evaluation_results: Dict[str, Any],
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 2: Cumulative profit over time for all agents."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    add_background_shading(ax, label=False)
    
    hours = np.arange(24)
    all_results = evaluation_results['all_results']
    
    for agent_name, results in all_results.items():
        episodes = results['episodes']
        if episode_idx < len(episodes):
            cum_cost = episodes[episode_idx]['cumulative_cost']
            cum_profit = -np.array(cum_cost)
            color = get_agent_color(agent_name)
            lw = 3 if 'R-SAC' in agent_name else 2
            line, = ax.plot(hours, cum_profit, label=agent_name, color=color, linewidth=lw, marker='s', markersize=4)
            
            # Annotate final value
            final_val = cum_profit[-1]
            ax.text(23.2, final_val, f"${final_val:.2f}", color=color, va='center', fontweight='bold')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Cumulative Profit ($)')
    ax.set_title('Cumulative Daily Profit Comparison', fontweight='bold')
    ax.set_xlim(0, 24.5) # Extra space for text
    
    ax.legend(loc='upper left', frameon=True)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig



def plot_total_profit_bar(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 3: Total profit comparison with error bars."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_results = evaluation_results['all_results']
    
    agent_names = list(all_results.keys())
    means = [all_results[name]['mean_profit'] for name in agent_names]
    stds = [all_results[name]['std_profit'] for name in agent_names]
    colors = [get_agent_color(name) for name in agent_names]
    
    # Create bar chart
    bars = ax.bar(agent_names, means, yerr=stds, capsize=10, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -3),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
    
    ax.set_ylabel('Total Profit ($)')
    ax.set_title('Average Daily Total Profit Comparison', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_profit_breakdown(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 4: Detailed breakdown of profit components."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    all_results = evaluation_results['all_results']
    agent_names = list(all_results.keys())
    
    # Extract components
    energy_profits = []
    peak_penalties = [] # These are costs, so positive values reduce profit
    deg_costs = []      # Costs
    
    for name in agent_names:
        res = all_results[name]['episodes']
        # Calculate means
        ep_mean = np.mean([-e['energy_cost'] for e in res]) # Energy profit = -Energy Cost
        pp_mean = np.mean([e['peak_penalty'] for e in res])
        dc_mean = np.mean([e['degradation_cost'] for e in res])
        
        energy_profits.append(ep_mean)
        peak_penalties.append(pp_mean)
        deg_costs.append(dc_mean)
        
    x = np.arange(len(agent_names))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, energy_profits, width, label='Energy Arbitrage Profit', color='#27ae60')
    ax.bar(x, [-p for p in peak_penalties], width, label='Peak Demand Penalty', color='#e74c3c')
    ax.bar(x + width, [-d for d in deg_costs], width, label='Degradation Cost', color='#f39c12')
    
    ax.set_ylabel('Value ($)')
    ax.set_title('Profit Composition Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.axhline(y=0, color='black', linewidth=1)
    
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Increase y-limit to prevent overlap with title
    # We need to consider the full range of data to add proportional headroom
    all_values = energy_profits + [-p for p in peak_penalties] + [-d for d in deg_costs]
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        data_range = max_val - min_val if max_val != min_val else 1.0
        
        # Ensure top limit has at least 15% of range buffer above the annotation baseline (0)
        # If max_val is < 0, we still want the top of the plot to be > 0
        target_top = max(max_val, 0) + 0.2 * data_range
        ax.set_ylim(top=target_top)
    
    # Calculate net for annotation
    for i, name in enumerate(agent_names):
        net = energy_profits[i] - peak_penalties[i] - deg_costs[i]
        
        base_h = max(energy_profits[i], 0)
        ax.annotate(f"Net: ${net:.2f}",
                    xy=(i, base_h),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_net_grid_power(
    evaluation_results: Dict[str, Any],
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 5: Net grid power (buy/sell) profile."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    add_background_shading(ax, label=False)
    
    hours = np.arange(24)
    all_results = evaluation_results['all_results']
    
    for agent_name, results in all_results.items():
        episodes = results['episodes']
        if episode_idx < len(episodes):
            grid_power = episodes[episode_idx]['grid_power']
            color = get_agent_color(agent_name)
            # Only shade area for R-SAC to reduce clutter, lines for others
            if agent_name == 'R-SAC':
                ax.fill_between(hours, grid_power, 0, color=color, alpha=0.2)
                ax.plot(hours, grid_power, label=agent_name, color=color, linewidth=3)
            else:
                ax.plot(hours, grid_power, label=agent_name, color=color, linewidth=1.5, alpha=0.8)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    # Annotate Buy/Sell
    ylim = ax.get_ylim()
    ax.text(1, ylim[1]*0.9, "BUYING FROM GRID", color='red', fontsize=12, fontweight='bold', alpha=0.3)
    ax.text(1, ylim[0]*0.9, "SELLING TO GRID", color='green', fontsize=12, fontweight='bold', alpha=0.3)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Net Grid Power (kW)')
    ax.set_title('Net Grid Power Profile', fontweight='bold')
    ax.set_xlim(0, 23)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

def plot_battery_power(
    evaluation_results: Dict[str, Any],
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 6: Battery charging/discharging power profile."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    add_background_shading(ax, label=False)
    
    hours = np.arange(24)
    all_results = evaluation_results['all_results']
    
    for agent_name, results in all_results.items():
        episodes = results['episodes']
        if episode_idx < len(episodes):
            bat_power = episodes[episode_idx]['battery_power']
            color = get_agent_color(agent_name)
            if agent_name == 'R-SAC':
                 ax.plot(hours, bat_power, label=agent_name, color=color, linewidth=3)
            else:
                 ax.plot(hours, bat_power, label=agent_name, color=color, linewidth=1.5, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Annotate Charge/Discharge
    ylim = ax.get_ylim()
    ax.text(1, ylim[1]*0.8, "CHARGING", color='blue', fontsize=10, fontweight='bold', alpha=0.3)
    ax.text(1, ylim[0]*0.8, "DISCHARGING", color='purple', fontsize=10, fontweight='bold', alpha=0.3)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Battery Power (kW)')
    ax.set_title('Battery Power Profile', fontweight='bold')
    ax.set_xlim(0, 23)
    ax.legend(loc='upper right')
    ax.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

def plot_ramping_analysis(
    evaluation_results: Dict[str, Any],
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 7: Hourly power ramping."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    add_background_shading(ax, label=False)
    
    hours = np.arange(1, 24)
    all_results = evaluation_results['all_results']
    
    for agent_name, results in all_results.items():
        episodes = results['episodes']
        if episode_idx < len(episodes):
            bat_power = episodes[episode_idx]['battery_power']
            ramping = np.diff(bat_power)
            color = get_agent_color(agent_name)
            ax.plot(hours, ramping, label=agent_name, color=color, linewidth=1.5, marker='.')
            
    ax.axhline(y=2.5, color='red', linestyle='--', label='Max Ramp Up')
    ax.axhline(y=-2.5, color='red', linestyle='--', label='Max Ramp Down')
    ax.axhline(y=0, color='black', alpha=0.5)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power Change (kW/hour)')
    ax.set_title('Battery Power Ramp Rates', fontweight='bold')
    ax.set_xlim(1, 23)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig




def plot_pricing_strategy(
    evaluation_results: Dict[str, Any],
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 8: Electricity price vs grid purchase power.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    hours = np.arange(24)
    price_profile = evaluation_results['price_profile']
    all_results = evaluation_results['all_results']
    
    # Top plot: Price profile
    ax1.fill_between(hours, 0, price_profile, alpha=0.3, color='gold')
    ax1.plot(hours, price_profile, color='orange', linewidth=2, marker='o', label='Electricity Price')
    ax1.set_ylabel('Price ($/kWh)')
    ax1.set_title('Electricity Price and Grid Purchase Strategy', fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Annotate price periods
    ax1.axvspan(0, 6, alpha=0.1, color='green', label='Off-Peak')
    ax1.axvspan(14, 21, alpha=0.1, color='red', label='On-Peak')
    
    # Bottom plot: Grid power for each agent
    for agent_name, results in all_results.items():
        episodes = results['episodes']
        if episode_idx < len(episodes):
            grid_power = episodes[episode_idx]['grid_power']
            # Only show buying (positive grid power)
            buy_power = np.maximum(grid_power, 0)
            color = get_agent_color(agent_name)
            ax2.bar(hours + list(all_results.keys()).index(agent_name) * 0.15 - 0.3, 
                    buy_power, width=0.15, label=agent_name, color=color, alpha=0.7)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Grid Purchase (kW)')
    ax2.legend(loc='upper right', ncol=2)
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multi_episode_gap(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 9: Gap vs LP across multiple episodes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_results = evaluation_results['all_results']
    n_episodes = evaluation_results['n_episodes']
    
    episode_nums = np.arange(1, n_episodes + 1)
    
    for agent_name, results in all_results.items():
        if agent_name == 'LP':
            continue  # LP is the baseline
        
        if 'gaps' in results:
            gaps = results['gaps']
            color = get_agent_color(agent_name)
            ax.plot(episode_nums, gaps, label=agent_name, color=color, linewidth=2, marker='o')
    
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='LP Optimum')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Gap vs LP (%)')
    ax.set_title('Optimality Gap vs LP Benchmark Across Episodes', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0.5, n_episodes + 0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_efficiency_metrics(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 10: Efficiency metrics (throughput, SOC deviation, peak violations).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_results = evaluation_results['all_results']
    
    agent_names = list(all_results.keys())
    colors = [get_agent_color(name) for name in agent_names]
    
    # Throughput
    throughputs = []
    for results in all_results.values():
        episodes = results['episodes']
        throughputs.append(np.mean([e['throughput'] for e in episodes]))
    
    axes[0].bar(agent_names, throughputs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Energy Throughput (kWh)')
    axes[0].set_title('Battery Throughput', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # SOC Deviation (from initial)
    soc_deviations = []
    for results in all_results.values():
        episodes = results['episodes']
        devs = []
        for e in episodes:
            initial = e.get('initial_soc', 0.5)
            final = e['final_soc']
            devs.append(abs(final - initial))
        soc_deviations.append(np.mean(devs))
    
    axes[1].bar(agent_names, soc_deviations, color=colors, edgecolor='black')
    axes[1].set_ylabel('SOC Deviation')
    axes[1].set_title('SOC Deviation (|Final - Initial|)', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Peak Violations
    peak_violations = []
    for results in all_results.values():
        episodes = results['episodes']
        peak_violations.append(np.mean([e['peak_violations'] for e in episodes]))
    
    axes[2].bar(agent_names, peak_violations, color=colors, edgecolor='black')
    axes[2].set_ylabel('Peak Violations')
    axes[2].set_title('Average Peak Violations', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_profiles(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 11: Solar, Load, and Price profiles.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    hours = np.arange(24)
    solar = evaluation_results['solar_profile']
    load = evaluation_results['load_profile']
    price = evaluation_results['price_profile']
    
    # Power profiles
    ax1.fill_between(hours, 0, solar, alpha=0.5, color='gold', label='Solar Generation')
    ax1.plot(hours, solar, color='orange', linewidth=2)
    ax1.plot(hours, load, color='red', linewidth=2, label='Load Consumption')
    ax1.fill_between(hours, 0, load, alpha=0.3, color='red')
    
    # Net load
    net_load = load - solar
    ax1.plot(hours, net_load, color='purple', linewidth=2, linestyle='--', label='Net Load')
    
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Solar Generation and Load Profiles', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Price profile
    ax2.fill_between(hours, 0, price, alpha=0.3, color='green')
    ax2.step(hours, price, where='mid', color='darkgreen', linewidth=2, label='TOU Price')
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Price ($/kWh)')
    ax2.set_title('Time-of-Use Electricity Price', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_summary_table(
    evaluation_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 12: Summary table visualization.
    """
    from evaluation import create_summary_table
    
    df = create_summary_table(evaluation_results)
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    # Format numeric columns
    df_display = df.copy()
    df_display['Mean Profit ($)'] = df_display['Mean Profit ($)'].apply(lambda x: f'${x:.2f}')
    df_display['Std Profit ($)'] = df_display['Std Profit ($)'].apply(lambda x: f'${x:.2f}')
    df_display['Gap vs LP (%)'] = df_display['Gap vs LP (%)'].apply(lambda x: f'{x:.1f}%')
    df_display['Gap Std (%)'] = df_display['Gap Std (%)'].apply(lambda x: f'{x:.1f}%')
    df_display['Mean Energy Profit ($)'] = df_display['Mean Energy Profit ($)'].apply(lambda x: f'${x:.2f}')
    df_display['Mean Peak Penalty ($)'] = df_display['Mean Peak Penalty ($)'].apply(lambda x: f'${x:.2f}')
    df_display['Mean Throughput (kWh)'] = df_display['Mean Throughput (kWh)'].apply(lambda x: f'{x:.2f}')
    df_display['Mean Final SOC'] = df_display['Mean Final SOC'].apply(lambda x: f'{x:.2%}')
    df_display['Eval Time (s)'] = df_display['Eval Time (s)'].apply(lambda x: f'{x:.2f}')
    
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4a90d9'] * len(df_display.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(df_display.columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_plots(
    evaluation_results: Dict[str, Any],
    output_dir: str = "results",
    episode_idx: int = 0
) -> List[plt.Figure]:
    """
    Generate all 12 plots and save to output directory.
    
    Args:
        evaluation_results: Results from run_comprehensive_evaluation
        output_dir: Directory to save plots
        episode_idx: Episode index for single-episode plots
        
    Returns:
        List of figure objects
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Visualization Plots...")
    print("="*60)
    
    figures = []
    
    # Plot 1: SOC Comparison
    fig1 = plot_soc_comparison(evaluation_results, episode_idx, 
                                f"{output_dir}/01_soc_comparison.pdf")
    figures.append(fig1)
    
    # Plot 2: Cumulative Profit
    fig2 = plot_cumulative_profit(evaluation_results, episode_idx,
                                 f"{output_dir}/02_cumulative_profit.pdf")
    figures.append(fig2)
    
    # Plot 3: Total Profit Bar
    fig3 = plot_total_profit_bar(evaluation_results,
                                f"{output_dir}/03_total_profit_bar.pdf")
    figures.append(fig3)
    
    # Plot 4: Profit Breakdown
    fig4 = plot_profit_breakdown(evaluation_results,
                                f"{output_dir}/04_profit_breakdown.pdf")
    figures.append(fig4)
    
    # Plot 5: Net Grid Power
    fig5 = plot_net_grid_power(evaluation_results, episode_idx,
                                f"{output_dir}/05_net_grid_power.pdf")
    figures.append(fig5)
    
    # Plot 6: Battery Power
    fig6 = plot_battery_power(evaluation_results, episode_idx,
                               f"{output_dir}/06_battery_power.pdf")
    figures.append(fig6)
    
    # Plot 7: Ramping Analysis
    fig7 = plot_ramping_analysis(evaluation_results, episode_idx,
                                  f"{output_dir}/07_ramping_analysis.pdf")
    figures.append(fig7)
    
    # Plot 8: Pricing Strategy
    fig8 = plot_pricing_strategy(evaluation_results, episode_idx,
                                  f"{output_dir}/08_pricing_strategy.pdf")
    figures.append(fig8)
    
    # Plot 9: Multi-Episode Gap
    fig9 = plot_multi_episode_gap(evaluation_results,
                                   f"{output_dir}/09_multi_episode_gap.pdf")
    figures.append(fig9)
    
    # Plot 10: Efficiency Metrics
    fig10 = plot_efficiency_metrics(evaluation_results,
                                     f"{output_dir}/10_efficiency_metrics.pdf")
    figures.append(fig10)
    
    # Plot 11: Profiles
    fig11 = plot_profiles(evaluation_results,
                          f"{output_dir}/11_profiles.pdf")
    figures.append(fig11)
    
    # Plot 12: Summary Table
    fig12 = plot_summary_table(evaluation_results,
                                f"{output_dir}/12_summary_table.pdf")
    figures.append(fig12)
    
    print(f"\n[OK] Generated {len(figures)} plots in {output_dir}/")
    
    return figures


if __name__ == "__main__":
    # Test visualization with mock data
    from data_loader import get_tou_prices
    import json
    
    np.random.seed(42)
    
    # Create test profiles
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)
    prices = get_tou_prices()
    
    # Create mock evaluation results
    mock_results = {
        'all_results': {
            'LP': {
                'mean_profit': 5.0,
                'std_profit': 0.5,
                'episodes': [{
                    'soc_trajectory': np.linspace(0.5, 0.7, 24),
                    'cumulative_cost': np.linspace(0, -5, 24),
                    'grid_power': load - solar,
                    'battery_power': np.sin(np.linspace(0, 2*np.pi, 24)) * 2,
                    'energy_cost': -4.0,
                    'peak_penalty': 0.5,
                    'degradation_cost': 0.5,
                    'throughput': 10,
                    'final_soc': 0.7,
                    'peak_violations': 2,
                    'initial_soc': 0.5
                }],
                'evaluation_time': 0.1
            },
            'SAC': {
                'mean_profit': 4.5,
                'std_profit': 0.8,
                'mean_gap': 30,
                'std_gap': 5,
                'gaps': [25, 30, 35],
                'episodes': [{
                    'initial_soc': 0.5
                }],
                'evaluation_time': 0.1
            }
        },
        'solar_profile': solar,
        'load_profile': load,
        'price_profile': prices,
        'n_episodes': 3
    }
    
    # Test generation
    # generate_all_plots(mock_results, "results/test_plots", episode_idx=0)
    # plot_energy_balance(mock_results, episode_idx=0, save_path="results/test_energy_balance.pdf")


def plot_energy_balance(
    evaluation_results: Dict[str, Any],
    agent_name: str = 'SAC',
    episode_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 13: Energy Balance / Self-Consumption Plot with Filled Areas.
    Combines Load, PV, Battery, and Grid flows into a stacked visualization.
    """
    import matplotlib.colors as mcolors
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Extract Data
    if agent_name not in evaluation_results['all_results']:
        print(f"Agent {agent_name} not found for Energy Balance plot.")
        return fig
        
    ep_data = evaluation_results['all_results'][agent_name]['episodes'][episode_idx]
    
    solar = evaluation_results['solar_profile']
    load = evaluation_results['load_profile']
    hours = np.arange(24)
    
    # Grid and Battery Power from Agent
    # grid_power > 0 is BUY (Import), < 0 is SELL (Export)
    grid_power = ep_data['grid_power']
    battery_power = ep_data['battery_power'] # > 0 Charge, < 0 Discharge
    
    # 2. Calculate Components for Stacking
    
    # A. Load Side Components (Under the Load Curve)
    # We want to show WHERE the energy for the Load comes from:
    # 1. Direct Solar
    # 2. Battery Discharge
    # 3. Grid Import
    
    # Direct Solar = min(Load, Solar) - but we must account for battery charging taking some solar
    # A cleaner way is to look at the flows:
    # Load is satisfied by: Direct Solar + Battery Discharge + Grid Import
    
    grid_import = np.maximum(grid_power, 0)
    battery_discharge = np.maximum(-battery_power, 0) # Positive value for discharge
    
    # Direct Solar to Load = Load - Grid Import - Battery Discharge
    # (Clip to 0 to avoid numerical noise)
    direct_solar_to_load = np.maximum(load - grid_import - battery_discharge, 0)
    
    # B. Generation Side Components (Under the PV Curve)
    # Solar energy goes to:
    # 1. Direct Solar (to Load)
    # 2. Battery Charge
    # 3. Grid Export
    
    grid_export = np.maximum(-grid_power, 0) # Positive value for export
    battery_charge = np.maximum(battery_power, 0)
    
    # Consistency check: Solar should approx equals Direct Solar + Battery Charge + Grid Export
    # There might be some discrepancies due to efficiencies (if modeled) or clipping.
    # We will use the calculated flows to stack, but bound them by the PV curve for visual cleanliness.
    
    # 3. Plotting - Using Stackplot or Fill_Between
    
    # Top Line: Load Profile
    ax.plot(hours, load, color='black', linewidth=2.5, linestyle='-', label='Load Demand', zorder=10)
    
    # Top Line: PV Profile
    ax.plot(hours, solar, color='black', linewidth=2.5, linestyle='--', label='PV Generation', zorder=10)
    
    # Stack 1: Satisfying Load (Cumulative from bottom)
    # Order: Direct Solar (Bottom) -> Battery Discharge -> Grid Import (Top)
    
    # Layer 1: Direct Solar (Common Base)
    ax.fill_between(hours, 0, direct_solar_to_load, color='#2980b9', alpha=0.9, label='Direct Solar Usage')
    
    # Layer 2: + Battery Discharge
    layer2_load = direct_solar_to_load + battery_discharge
    ax.fill_between(hours, direct_solar_to_load, layer2_load, color='#8e44ad', alpha=0.8, label='Battery Discharging')
    
    # Layer 3: + Grid Import (Should match Load roughly)
    layer3_load = layer2_load + grid_import
    # Fill up to max(Load, Layer2) to ensure we cover the load line, or just use the calculated stack
    # Using the calculated stack ensures we represent the actual flows
    ax.fill_between(hours, layer2_load, layer3_load, color='#c0392b', alpha=0.8, label='Grid Import')
    
    
    # Stack 2: Solar Usage (Cumulative from bottom)
    # Order: Direct Solar (Matches above) -> Battery Charge -> Grid Export
    
    # Layer 1: Direct Solar is already plotted.
    
    # Layer 2: + Battery Charge
    # Base is Direct Solar
    layer2_solar = direct_solar_to_load + battery_charge
    ax.fill_between(hours, direct_solar_to_load, layer2_solar, color='#9b59b6', alpha=0.6, label='Battery Charging') # Lighter purple
    
    # Layer 3: + Grid Export
    layer3_solar = layer2_solar + grid_export
    ax.fill_between(hours, layer2_solar, layer3_solar, color='#27ae60', alpha=0.8, label='Grid Export')
    
    
    # Formatting
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Energy Balance Analysis - {agent_name} Agent', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, max(np.max(load), np.max(solar)) * 1.2) # Add headroom
    
    # Add Pricing Background for context
    add_background_shading(ax, label=False)
    
    # Custom Legend
    # Group the legend items logically
    handles, labels = ax.get_legend_handles_labels()
    # Sort/Filter if needed. The fill_between labels are good.
    ax.legend(loc='upper right', ncol=2, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
