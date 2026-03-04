import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_duck_curve():
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    
    # Time axis
    hours = np.linspace(0, 24, 240)
    
    # Generate synthetic profiles for illustration
    # Typical residential load: Morning peak, low mid-day, high evening peak
    load_base = 3.0
    load_morning = 2.0 * np.exp(-((hours - 8)**2) / 4)
    load_evening = 4.0 * np.exp(-((hours - 19)**2) / 8)
    load = load_base + load_morning + load_evening
    
    # Solar profile: Bell curve centered at noon
    solar_peak = 6.0
    solar = solar_peak * np.exp(-((hours - 12)**2) / 6)
    solar = np.clip(solar, 0, None)
    
    # Net Load = Load - Solar
    net_load = load - solar
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines with explicit styling
    ax.plot(hours, load, label='Total Load (Scenario)', color='#e74c3c', linewidth=3, linestyle='--')
    ax.plot(hours, solar, label='Solar PV Generation', color='#f1c40f', linewidth=3, alpha=0.8)
    ax.plot(hours, net_load, label='Net Load (Duck Curve)', color='#2980b9', linewidth=4)
    
    # Highlight the "Belly"
    belly_idx = np.argmin(net_load)
    belly_hour = hours[belly_idx]
    belly_val = net_load[belly_idx]
    
    ax.annotate('Overgeneration Risk\n(The "Belly")', 
                xy=(belly_hour, belly_val), 
                xytext=(belly_hour, belly_val - 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', fontsize=12)

    # Highlight the "Ramp"
    ramp_hour = 17
    ramp_high = 20
    # Find values at these hours
    idx_17 = np.abs(hours - 17).argmin()
    idx_20 = np.abs(hours - 20).argmin()
    
    # Draw arrow for ramp
    # ax.annotate('', xy=(20, net_load[idx_20]), xytext=(17, net_load[idx_17]),
    #             arrowprops=dict(facecolor='red', arrowstyle='simple', alpha=0.5))
    
    ax.text(18.5, (net_load[idx_17] + net_load[idx_20])/2, 'Ramping Need', 
            rotation=60, color='#c0392b', fontweight='bold', ha='right')

    # Shade the area between load and net load (Solar contribution)
    ax.fill_between(hours, load, net_load, color='#f1c40f', alpha=0.15)
    
    # Formatting
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Power (kW)', fontsize=14)
    ax.set_title('The "Duck Curve" Phenomenon', fontsize=16, pad=15)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig('Pages/fig/duck_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Duck curve generated successfully at Pages/fig/duck_curve.pdf")

if __name__ == "__main__":
    generate_duck_curve()
