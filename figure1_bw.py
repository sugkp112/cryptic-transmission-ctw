"""
REVISED Figure 1: Decoupling Phenomenon
BLACK-WHITE OPTIMIZED:
- Panel A: Black solid (infection) vs Grey dashed+markers (beetle)
- Panel B: Black solid timing metric, grey dashed threshold
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0

np.random.seed(42)
time_months = 48
N_trees = 1000

# Model parameters
lambda_beetle = 0.02
beta_infection = 0.7
sigma_latent = 0.05
mu_disease = 0.03
alpha_reinforcement = 2.5

# Initialize and simulate
S = np.zeros(time_months)
E = np.zeros(time_months)
I = np.zeros(time_months)
D = np.zeros(time_months)
B = np.zeros(time_months)

S[0] = N_trees

for t in range(1, time_months):
    s, e, i, d = S[t-1], E[t-1], I[t-1], D[t-1]
    
    reinforcement = 1 + alpha_reinforcement * (e + i) / N_trees
    beetle_events = np.random.poisson(lambda_beetle * s * reinforcement)
    B[t] = max(0, beetle_events * 0.15 + np.random.normal(0, 0.5))
    
    new_infections = np.random.binomial(int(beetle_events), beta_infection)
    e_to_i = np.random.binomial(int(e), sigma_latent)
    i_to_d = np.random.binomial(int(i), mu_disease)
    
    S[t] = s - new_infections
    E[t] = e + new_infections - e_to_i
    I[t] = i + e_to_i - i_to_d
    D[t] = d + i_to_d

timing_metric = (E + I) / (B + 0.1)
infected_prop = (E + I) / N_trees * 100
time_years = np.arange(time_months) / 12

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))

# ============================================================================
# Panel A: Infection vs Beetle (OPTIMIZED FOR B/W)
# ============================================================================

# Infected trees: BLACK SOLID, NO MARKER
ax1.plot(time_years, infected_prop, linewidth=2.5, color='black', 
         label='Infected trees (%)', linestyle='-')

ax1.set_ylabel('Infected trees (%)', fontsize=11, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, 70)

# Beetle activity: DARK GREY DASHED + CIRCLE MARKERS
ax1_twin = ax1.twinx()
ax1_twin.plot(time_years, B, linewidth=2, color='0.4',  # grey = 0.4
              linestyle='--', marker='o', markevery=6, markersize=5,
              label='Beetle detections')
ax1_twin.set_ylabel('Beetle detections (per month)', fontsize=11, color='0.4')
ax1_twin.tick_params(axis='y', labelcolor='0.4')
ax1_twin.set_ylim(0, 7.0)  # Increased to accommodate data peaks

# Decoupling marker
decoupling_time = np.argmax(np.diff(infected_prop) > 5) / 12
if decoupling_time > 0:
    ax1.axvline(x=decoupling_time, color='black', linestyle=':', 
                linewidth=1, alpha=0.6)

ax1.set_xlabel('Time (years)', fontsize=11)
ax1.grid(True, alpha=0.2, linestyle=':')
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
           fontsize=9, framealpha=0.9)

# ============================================================================
# Panel B: Timing-based metric (BLACK SOLID + GREY DASHED THRESHOLD)
# ============================================================================

# Timing metric: BLACK SOLID, NO MARKER
ax2.plot(time_years, timing_metric, linewidth=2.5, color='black', linestyle='-')

# Threshold: GREY DASHED
ax2.axhline(y=5, color='0.6', linestyle='--', linewidth=1.5, 
            label='Detection threshold')

ax2.set_xlabel('Time (years)', fontsize=11)
ax2.set_ylabel('Pre-detection transmission metric (a.u.)', fontsize=11)
ax2.set_ylim(0, 350)  # Fixed upper limit to accommodate late-stage spikes
ax2.grid(True, alpha=0.2, linestyle=':')
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('Fig1_Decoupling_Phenomenon_REVISED_BW.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig1_Decoupling_Phenomenon_REVISED_BW.pdf', bbox_inches='tight')
print("Figure 1 saved (B/W optimized - linestyle + marker distinction)")
plt.close()