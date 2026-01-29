"""
REVISED Figure 2: Phase Diagram
BLACK-WHITE OPTIMIZED:
- Keep grayscale colormap
- Only 2-3 key contours with black lines
- Bold threshold=5 contour
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

# Parameter space
lambda_range = np.linspace(0.001, 0.1, 50)
pathogen_ratio_range = np.linspace(0.5, 10, 50)

N_trees = 500
time_steps = 80
sigma_base = 0.05
alpha_reinforcement = 2.0

# Results matrices
final_infection = np.zeros((len(pathogen_ratio_range), len(lambda_range)))
max_timing_metric = np.zeros((len(pathogen_ratio_range), len(lambda_range)))

print("Simulating parameter space...")
for i, pathogen_ratio in enumerate(pathogen_ratio_range):
    for j, lambda_val in enumerate(lambda_range):
        beta = pathogen_ratio * sigma_base
        
        S, E, I = N_trees, 0, 0
        B_avg = 0
        
        for t in range(time_steps):
            reinforcement = 1 + alpha_reinforcement * (E + I) / N_trees
            beetle_events = max(0, np.random.poisson(lambda_val * S * reinforcement))
            B_visible = beetle_events * 0.15
            B_avg += B_visible
            
            new_inf = min(S, int(beetle_events * min(beta, 1.0)))
            e_to_i = int(E * sigma_base)
            
            S -= new_inf
            E += new_inf - e_to_i
            I += e_to_i
        
        final_infection[i, j] = (E + I) / N_trees * 100
        B_avg = B_avg / time_steps
        max_timing_metric[i, j] = (E + I) / (B_avg + 0.1) if B_avg > 0.01 else 0

final_infection_smooth = gaussian_filter(final_infection, sigma=1.0)
max_timing_metric_smooth = gaussian_filter(max_timing_metric, sigma=1.0)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ============================================================================
# Panel A: Final infection (2-3 KEY CONTOURS ONLY)
# ============================================================================

infection_cmap = LinearSegmentedColormap.from_list('infection',
    ['#F0F0F0', '#D3D3D3', '#A9A9A9', '#808080', '#696969', 
     '#CD5C5C', '#8B0000', '#6B0000'])

im1 = ax1.contourf(lambda_range * 100, pathogen_ratio_range, 
                   final_infection_smooth, levels=20, cmap=infection_cmap)

# ONLY 2 KEY CONTOURS: 30% and 50%
contour1 = ax1.contour(lambda_range * 100, pathogen_ratio_range, 
                       final_infection_smooth, levels=[30, 50], 
                       colors='black', linewidths=1.2, linestyles='-')
ax1.clabel(contour1, inline=True, fontsize=8, fmt='%1.0f')

ax1.set_xlabel('Beetle arrival rate (λ × 100)', fontsize=11)
ax1.set_ylabel('Pathogen establishment ratio (β/σ)', fontsize=11)
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Final infected trees (%)', fontsize=10)

# ============================================================================
# Panel B: Timing metric (BOLD threshold=5, normal for others)
# ============================================================================

cii_cmap = LinearSegmentedColormap.from_list('timing',
    ['#FFFFFF', '#E0E0E0', '#C0C0C0', '#A0A0A0', '#808080', '#606060', '#404040'])

masked_metric = np.ma.masked_where(max_timing_metric_smooth < 3, max_timing_metric_smooth)

im2 = ax2.contourf(lambda_range * 100, pathogen_ratio_range, 
                   masked_metric, levels=15, cmap=cii_cmap, extend='max')

# KEY CONTOURS with threshold=5 EMPHASIZED
contour2 = ax2.contour(lambda_range * 100, pathogen_ratio_range, 
                       max_timing_metric_smooth, levels=[5, 10, 20], 
                       colors='black', 
                       linewidths=[1.8, 1.2, 1.2],  # threshold=5 is 30% bolder
                       linestyles='-')
ax2.clabel(contour2, inline=True, fontsize=8, fmt='%1.0f')

ax2.set_xlabel('Beetle arrival rate (λ × 100)', fontsize=11)
ax2.set_ylabel('Pathogen establishment ratio (β/σ)', fontsize=11)
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Pre-detection transmission (a.u.)', fontsize=10)

plt.tight_layout()
plt.savefig('Fig2_Phase_Diagram_REVISED_BW.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig2_Phase_Diagram_REVISED_BW.pdf', bbox_inches='tight')
print("Figure 2 saved (B/W optimized - minimal contours)")
plt.close()