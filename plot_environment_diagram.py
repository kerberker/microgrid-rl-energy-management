
"""
Plot Environment Architecture Diagram
Generates a publication-quality PDF diagram of the Microgrid System Architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_diagram(save_path="results/environment_architecture.pdf"):
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # --- 1. Background Layers ---
    
    # Cyber Layer (Top) - Light Blue
    cyber_rect = patches.Rectangle((0, 4.5), 10, 2.5, facecolor='#d6eaf8', edgecolor='#3498db', alpha=0.5, zorder=0)
    ax.add_patch(cyber_rect)
    ax.text(1.5, 6, "CYBER LAYER\n(Control/Agent)", ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # Physical Layer (Bottom) - Light Green
    phys_rect = patches.Rectangle((0, 0), 10, 4.2, facecolor='#d5f5e3', edgecolor='#27ae60', alpha=0.5, zorder=0)
    ax.add_patch(phys_rect)
    ax.text(1.5, 3.5, "PHYSICAL LAYER\n(Microgrid)", ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # --- 2. Components (Boxes) ---
    
    # Helper to draw box with centered text
    def draw_box(x, y, w, h, text, color, edge_color='black'):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge_color, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
        return (x + w/2, y + h/2) # Return center
        
    # Agent (In Cyber Layer)
    agent_x, agent_y = 4.0, 5.0
    agent_w, agent_h = 3.0, 1.2
    agent_center = draw_box(agent_x, agent_y, agent_w, agent_h, "RL Agent\n(EMS)", '#a9cce3')
    
    # Physical Components
    comp_w, comp_h = 1.8, 1.0
    
    # Grid (Top Left)
    grid_x, grid_y = 2.5, 2.5
    grid_center = draw_box(grid_x, grid_y, comp_w, comp_h, "Grid\nConnection", '#d7dbdd') # Grey
    
    # Solar (Top Right)
    solar_x, solar_y = 6.5, 2.5
    solar_center = draw_box(solar_x, solar_y, comp_w, comp_h, "Solar PV", '#f9e79f', '#f1c40f') # Yellow
    
    # Load (Bottom Left)
    load_x, load_y = 2.5, 0.5
    load_center = draw_box(load_x, load_y, comp_w, comp_h, "House Load", '#fadbd8', '#e74c3c') # Red
    
    # Battery (Bottom Right)
    batt_x, batt_y = 6.5, 0.5
    batt_center = draw_box(batt_x, batt_y, comp_w, comp_h, "Battery Storage\n(ESS)", '#d2b4de', '#8e44ad') # Purple
    
    # DC/AC Bus (Center)
    bus_circle = patches.Circle((5.5, 1.75), 0.6, facecolor='#fAD7A0', edgecolor='#F39C12', linewidth=1.5, zorder=2)
    ax.add_patch(bus_circle)
    ax.text(5.5, 1.75, "DC/AC\nBus", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    bus_center = (5.5, 1.75)
    
    # --- 3. Connections ---
    
    # Style props for ArrowProps
    power_arrow_props = dict(arrowstyle='<->', linewidth=3, color='black', shrinkA=0, shrinkB=0)
    power_arrow_props_directed = dict(arrowstyle='->', linewidth=3, color='black', shrinkA=0, shrinkB=0)
    
    info_arrow_props = dict(arrowstyle='->', linewidth=1.5, color='#0055aa', linestyle='--', shrinkA=0, shrinkB=0)
    
    # --- Power Flows (Solid) ---
    # Grid <-> Bus
    ax.annotate("", xy=bus_center, xytext=(grid_x+comp_w, grid_y+comp_h/2), arrowprops=power_arrow_props, zorder=1)
    
    # Solar -> Bus
    # Draw manuall lines for Solar path to avoid complex annotation logic
    ax.plot([solar_x, 5.5], [solar_y+comp_h/2, solar_y+comp_h/2], lw=3, color='black', zorder=1) # Horizontal
    ax.plot([5.5, 5.5], [solar_y+comp_h/2, 1.75], lw=3, color='black', zorder=1) # Vertical
    # Arrow head at Bus
    ax.annotate("", xy=bus_center, xytext=(5.5, 3.0), arrowprops=power_arrow_props_directed, zorder=1)

    # Bus <-> Battery
    ax.annotate("", xy=(batt_x, batt_y+comp_h/2), xytext=bus_center, arrowprops=power_arrow_props, zorder=1)
    
    # Bus -> Load
    ax.annotate("", xy=(load_x+comp_w, load_y+comp_h/2), xytext=bus_center, arrowprops=power_arrow_props_directed, zorder=1)
    
    # Label: "Power Flow"
    ax.text(4.5, 2.0, "Power Flow", fontsize=10, rotation=35, ha='center')

    # --- Information Flows (Dashed) ---
    
    # Agent Input: Price (From Grid/Market)
    ax.plot([grid_x+comp_w/2, grid_x+comp_w/2], [grid_y+comp_h, agent_y+0.6], lw=1.5, color='#0055aa', linestyle='--', zorder=1)
    ax.plot([grid_x+comp_w/2, agent_x], [agent_y+0.6, agent_y+0.6], lw=1.5, color='#0055aa', linestyle='--', zorder=1)
    # Head
    ax.annotate("", xy=(agent_x, agent_y+0.6), xytext=(agent_x-0.1, agent_y+0.6), arrowprops=info_arrow_props)
    ax.text(3.1, 4.3, "Price", color='black', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Agent Input: Generation (From Solar)
    ax.plot([solar_x+comp_w/2, solar_x+comp_w/2], [solar_y+comp_h, agent_y], lw=1.5, color='#0055aa', linestyle='--', zorder=1)
    # Head at agent bottom
    ax.annotate("", xy=(solar_x+comp_w/2, agent_y), xytext=(solar_x+comp_w/2, agent_y-0.1), arrowprops=info_arrow_props)
    ax.text(7.0, 4.0, "Generation", color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Agent Input: SOC (From Battery)
    path_x = [batt_x+comp_w*0.8, batt_x+comp_w*0.8, 9.5, 9.5, agent_x+agent_w]
    path_y = [batt_y+comp_h, 4.0, 4.0, agent_y+0.4, agent_y+0.4]
    ax.plot(path_x, path_y, lw=1.5, color='#0055aa', linestyle='--', zorder=1)
    # Head
    ax.annotate("", xy=(agent_x+agent_w, agent_y+0.4), xytext=(agent_x+agent_w+0.1, agent_y+0.4), arrowprops=info_arrow_props)
    ax.text(9.0, 4.2, "SOC", color='black', fontsize=10, ha='center')
    
    # Agent Output: Charge/Discharge Cmd
    path_x = [agent_x+agent_w, 9.8, 9.8, batt_x+comp_w/2]
    path_y = [agent_y+0.8, agent_y+0.8, 1.8, batt_y+comp_h]
    ax.plot(path_x, path_y, lw=1.5, color='#0055aa', linestyle='--', zorder=1)
    # Head
    ax.annotate("", xy=(batt_x+comp_w/2, batt_y+comp_h), xytext=(batt_x+comp_w/2, batt_y+comp_h+0.1), arrowprops=info_arrow_props)
    ax.text(8.5, 6.0, "Charge/Discharge Cmd", color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    
    plt.tight_layout()
    
    # Create directory if needed
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Diagram saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    draw_diagram()
