"""
REVISED Figure 3: Management Strategy Comparison
BLACK-WHITE OPTIMIZED:
- Consistent linestyle + marker mapping across all panels
- Panel D: Use hatch patterns instead of colors
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

np.random.seed(123)

time_months = 60
N_trees = 1000
lambda_beetle = 0.025
beta_infection = 0.75
sigma_latent = 0.04
mu_disease = 0.03
alpha_reinforcement = 2.8

beetle_threshold = 2.0
timing_threshold = 2.5
sampling_rate = 0.04

scenarios = {
    'No intervention': {'intervene': False},
    'Beetle-based': {'beetle_trigger': True},
    'Timing-based early detection': {'timing_trigger': True}
}

results = {}

for scenario_name, config in scenarios.items():
    S = np.zeros(time_months)
    E = np.zeros(time_months)
    I = np.zeros(time_months)
    D = np.zeros(time_months)
    B = np.zeros(time_months)
    timing_metric = np.zeros(time_months)
    intervention_time = None
    intervention_active = False
    
    S[0] = N_trees
    
    for t in range(1, time_months):
        s, e, i, d = S[t-1], E[t-1], I[t-1], D[t-1]
        
        if not intervention_active:
            if config.get('beetle_trigger') and B[t-1] > beetle_threshold:
                intervention_active = True
                intervention_time = t
            elif config.get('timing_trigger'):
                if e + i > 0:
                    n_sampled = max(1, int(N_trees * sampling_rate))
                    detection_prob = min(1.0, (e + i) / N_trees * n_sampled / (e + i))
                    sampled_infected = np.random.binomial(int(e + i), detection_prob)
                    
                    if sampled_infected > 0:
                        estimated_infected = sampled_infected / sampling_rate
                        detected_metric = estimated_infected / (B[t-1] + 0.1)
                        
                        if detected_metric > timing_threshold or sampled_infected > 2:
                            intervention_active = True
                            intervention_time = t
        
        reinforcement = 1 + alpha_reinforcement * (e + i) / N_trees
        if intervention_active:
            lambda_eff = lambda_beetle * 0.25
            beta_eff = beta_infection * 0.4
        else:
            lambda_eff = lambda_beetle
            beta_eff = beta_infection
        
        beetle_events = np.random.poisson(lambda_eff * s * reinforcement)
        B[t] = max(0, beetle_events * 0.15 + np.random.normal(0, 0.3))
        
        new_infections = np.random.binomial(int(beetle_events), beta_eff)
        e_to_i = np.random.binomial(int(e), sigma_latent)
        i_to_d = np.random.binomial(int(i), mu_disease)
        
        S[t] = s - new_infections
        E[t] = e + new_infections - e_to_i
        I[t] = i + e_to_i - i_to_d
        D[t] = d + i_to_d
        
        timing_metric[t] = (E[t] + I[t]) / (B[t] + 0.1)
    
    results[scenario_name] = {
        'S': S, 'E': E, 'I': I, 'D': D, 'B': B, 'timing_metric': timing_metric,
        'intervention_time': intervention_time,
        'final_loss': (D[-1] / N_trees) * 100
    }

# Create figure with increased height
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                     height_ratios=[1.2, 1.2, 1.5],  # Make Panel D taller
                     left=0.1, right=0.95)  # Add margins to prevent label cutoff

time_years = np.arange(time_months) / 12

# CONSISTENT STYLE MAPPING (with distinct colors for Panel D bars)
styles = {
    'No intervention': {
        'color': '0.7',  # light grey
        'linestyle': ':',
        'marker': None,
        'markevery': None,
        'hatch': None,  # no hatch
        'bar_color': '0.85',  # very light grey for bars
        'bar_color_rgb': '#95a5a6'  # Grey - for visual distinction
    },
    'Beetle-based': {
        'color': '0.4',  # dark grey
        'linestyle': '--',
        'marker': 's',  # square
        'markevery': 8,
        'hatch': '///',  # triple slash (more visible)
        'bar_color': '0.6',  # medium grey for bars
        'bar_color_rgb': '#3498db'  # Blue - color blind friendly
    },
    'Timing-based early detection': {
        'color': 'black',
        'linestyle': '-',
        'marker': 'o',  # circle
        'markevery': 8,
        'hatch': '...',  # triple dots (more visible)
        'bar_color': '0.35',  # dark grey for bars
        'bar_color_rgb': '#e67e22'  # Orange - color blind friendly
    }
}

# ============================================================================
# Panel A: Infection dynamics
# ============================================================================

ax1 = fig.add_subplot(gs[0, :])

for scenario_name, data in results.items():
    style = styles[scenario_name]
    infected_prop = (data['E'] + data['I']) / N_trees * 100
    
    if style['marker']:
        ax1.plot(time_years, infected_prop, linewidth=2, 
                 color=style['color'], linestyle=style['linestyle'],
                 marker=style['marker'], markevery=style['markevery'],
                 markersize=5, label=scenario_name, alpha=0.9)
    else:
        ax1.plot(time_years, infected_prop, linewidth=2, 
                 color=style['color'], linestyle=style['linestyle'],
                 label=scenario_name, alpha=0.9)
    
    # Small marker at intervention
    if data['intervention_time']:
        t_int = data['intervention_time'] / 12
        inf_at_int = (data['E'][data['intervention_time']] + 
                      data['I'][data['intervention_time']]) / N_trees * 100
        ax1.plot([t_int], [inf_at_int], 'o', 
                color=style['color'], markersize=6)

ax1.set_xlabel('Time (years)', fontsize=11)
ax1.set_ylabel('Infected trees (%)', fontsize=11)
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.2, linestyle=':')
ax1.set_ylim(0, 70)

# ============================================================================
# Panel B: Beetle activity (SAME STYLE MAPPING)
# ============================================================================

ax2 = fig.add_subplot(gs[1, 0])

for scenario_name, data in results.items():
    style = styles[scenario_name]
    
    if style['marker']:
        ax2.plot(time_years, data['B'], linewidth=1.5, 
                 color=style['color'], linestyle=style['linestyle'],
                 marker=style['marker'], markevery=style['markevery'],
                 markersize=4, alpha=0.8)
    else:
        ax2.plot(time_years, data['B'], linewidth=1.5, 
                 color=style['color'], linestyle=style['linestyle'],
                 alpha=0.8)

# Threshold: black dotted
ax2.axhline(y=beetle_threshold, color='black', linestyle=':', 
            linewidth=1.2, alpha=0.6)

ax2.set_xlabel('Time (years)', fontsize=11)
ax2.set_ylabel('Beetle detections (per month)', fontsize=11)
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax2.set_ylim(0, 7.0)  # Explicit limit to accommodate initial spike
ax2.grid(True, alpha=0.2, linestyle=':')

# ============================================================================
# Panel C: Timing metric (emphasize timing-based, others faint)
# ============================================================================

ax3 = fig.add_subplot(gs[1, 1])

for scenario_name, data in results.items():
    style = styles[scenario_name]
    
    # Only show timing-based prominently
    if scenario_name == 'Timing-based early detection':
        ax3.plot(time_years, data['timing_metric'], linewidth=2, 
                 color='black', linestyle='-')
        # Mark threshold crossing
        if data['intervention_time']:
            t_int = data['intervention_time'] / 12
            ax3.plot([t_int], [data['timing_metric'][data['intervention_time']]], 
                    'o', color='black', markersize=6)
    else:
        # Others as faint grey dashed
        ax3.plot(time_years, data['timing_metric'], linewidth=1, 
                 color='0.7', linestyle='--', alpha=0.5)

ax3.axhline(y=timing_threshold, color='black', linestyle=':', 
            linewidth=1.2, alpha=0.6)

ax3.set_xlabel('Time (years)', fontsize=11)
ax3.set_ylabel('Derived pre-detection transmission metric (a.u.)', fontsize=11)
ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax3.grid(True, alpha=0.2, linestyle=':')
ax3.set_ylim(0, 1300)  # Set to 1300 to accommodate all extreme spikes

# ============================================================================
# Panel D: Outcomes (BAR CHART with hatch patterns for strategies)
# ============================================================================

ax4 = fig.add_subplot(gs[2, :])

scenario_names = list(results.keys())
final_losses = [results[s]['final_loss'] for s in scenario_names]
intervention_times = [results[s]['intervention_time'] if results[s]['intervention_time'] 
                     else time_months for s in scenario_names]

x = np.arange(len(scenario_names))
width = 0.35

# Define hatch patterns for each strategy
strategy_hatches = {
    'No intervention': None,      # No hatch (empty/white)
    'Beetle-based': '...',        # Dots
    'Timing-based early detection': '///'            # Diagonal lines
}

# Use neutral grey color for all, differentiate by hatch
base_color = '0.7'  # Medium grey

# Plot mortality bars (solid edge, darker fill)
bars1 = []
for i, scenario_name in enumerate(scenario_names):
    bar = ax4.bar(x[i] - width/2, final_losses[i], width,
                  color='0.6',  # Darker grey for mortality
                  edgecolor='black', linewidth=2.5,
                  linestyle='-', hatch=strategy_hatches[scenario_name])
    bars1.append(bar)

ax4_twin = ax4.twinx()

# Plot intervention time bars (dashed edge, lighter fill)
bars2 = []
for i, scenario_name in enumerate(scenario_names):
    bar = ax4_twin.bar(x[i] + width/2, intervention_times[i]/12, width,
                       color='0.85',  # Lighter grey for time
                       edgecolor='black', linewidth=2.5,
                       linestyle='--', hatch=strategy_hatches[scenario_name])
    bars2.append(bar)

# Value labels
for i, (bar, val) in enumerate(zip(bars1, final_losses)):
    height = bar[0].get_height()
    ax4.text(bar[0].get_x() + bar[0].get_width()/2., height + 0.8,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

for i, (bar, val) in enumerate(zip(bars2, intervention_times)):
    height = bar[0].get_height()
    scenario_name = scenario_names[i]
    if val < time_months:
        # Show actual intervention time
        ax4_twin.text(bar[0].get_x() + bar[0].get_width()/2., height + 0.15,
                      f'{val/12:.1f}y', ha='center', va='bottom', fontsize=9, 
                      fontweight='bold')
    else:
        # No intervention case - show N/A
        ax4_twin.text(bar[0].get_x() + bar[0].get_width()/2., height + 0.15,
                      'N/A', ha='center', va='bottom', fontsize=9, 
                      fontweight='bold', style='italic', color='0.4')

ax4.set_ylabel('Tree mortality (%)', fontsize=11, fontweight='bold')
ax4_twin.set_ylabel('Time to intervention (years)', fontsize=11, fontweight='bold')
ax4.set_title('D', fontsize=12, fontweight='bold', loc='left')
ax4.set_xticks(x)
ax4.set_xticklabels(scenario_names, fontsize=10)
ax4.set_ylim(0, 40)
ax4_twin.set_ylim(0, 6)
ax4.grid(True, alpha=0.2, linestyle=':', axis='y', zorder=0)

# Create comprehensive legend with linestyle and hatch differentiation
from matplotlib.patches import Patch, Rectangle
import matplotlib.lines as mlines

# First section: Metrics (distinguished by edge linestyle)
metrics_legend = [
    mlines.Line2D([], [], color='none', label='Metrics:', 
                 markerfacecolor='none', markeredgecolor='none'),
    Patch(facecolor='0.6', edgecolor='black', linewidth=2.5, 
          linestyle='-', label='  Solid edge = Tree mortality (%)'),
    Patch(facecolor='0.85', edgecolor='black', linewidth=2.5,
          linestyle='--', label='  Dashed edge = Intervention time (y)'),
]

# Second section: Strategies (distinguished by hatch patterns)
strategy_legend = [
    mlines.Line2D([], [], color='none', label=' ', 
                 markerfacecolor='none', markeredgecolor='none'),
    mlines.Line2D([], [], color='none', label='Strategies:', 
                 markerfacecolor='none', markeredgecolor='none'),
    Patch(facecolor='white', edgecolor='black', linewidth=2, 
          hatch=None, label='  Empty = No intervention'),
    Patch(facecolor='white', edgecolor='black', linewidth=2,
          hatch='...', label='  Dots = Beetle-based'),
    Patch(facecolor='white', edgecolor='black', linewidth=2,
          hatch='///', label='  Diagonal = Timing-based'),
]

# Combine into single legend
all_legend = metrics_legend + strategy_legend

ax4.legend(handles=all_legend, loc='upper right', fontsize=9,
          framealpha=0.95, handlelength=2.5, handleheight=1.8,
          borderpad=0.8, labelspacing=0.5)

plt.tight_layout()
plt.savefig('Fig3_Management_Comparison_REVISED_BW.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.savefig('Fig3_Management_Comparison_REVISED_BW.pdf', bbox_inches='tight', pad_inches=0.2)
print("Figure 3 saved (B/W optimized - consistent styles + hatch)")
print(f"  Beetle-based: {results['Beetle-based']['final_loss']:.2f}%, Timing-based: {results['Timing-based early detection']['final_loss']:.2f}%")
plt.close()