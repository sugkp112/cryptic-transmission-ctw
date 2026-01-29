"""
REVISED Supplementary Figure S3: Economic Analysis
BLACK-WHITE OPTIMIZED:
- Bars: Same grey, different hatch patterns
- ROI: Black solid + filled circles
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

np.random.seed(456)

sampling_rates = np.array([0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
n_replicates = 100
time_horizon = 48

# Economic parameters
tree_value = 500
sampling_cost = 50
intervention_cost = 10000
treatment_cost_per_tree = 30

N_trees = 1000
lambda_beetle = 0.025
beta_infection = 0.75
sigma_latent = 0.04
mu_disease = 0.03
alpha_reinforcement = 2.8

total_costs = np.zeros((len(sampling_rates), n_replicates))
mortality = np.zeros((len(sampling_rates), n_replicates))

print("Running economic analysis...")
for sr_idx, sampling_rate in enumerate(sampling_rates):
    for rep in range(n_replicates):
        S, E, I, D = N_trees, 0, 0, 0
        detected = False
        
        for t in range(time_horizon):
            reinforcement = 1 + alpha_reinforcement * (E + I) / N_trees
            beetle_events = np.random.poisson(lambda_beetle * S * reinforcement)
            new_inf = np.random.binomial(beetle_events, beta_infection)
            e_to_i = np.random.binomial(int(E), sigma_latent)
            i_to_d = np.random.binomial(int(I), mu_disease)
            
            S -= new_inf
            E += new_inf - e_to_i
            I += e_to_i - i_to_d
            D += i_to_d
            
            if not detected and sampling_rate > 0 and (E + I) > 0:
                n_sampled = max(1, int(N_trees * sampling_rate))
                if n_sampled >= (E + I):
                    sampled_infected = int(E + I)
                else:
                    sampled_infected = np.random.binomial(int(E + I), 
                                                          min(1.0, n_sampled / N_trees))
                
                if sampled_infected > 0:
                    detected = True
                    lambda_beetle *= 0.3
                    beta_infection *= 0.5
        
        surveillance_cost = sampling_cost * sampling_rate * N_trees * time_horizon
        if detected:
            intervention_cost_total = intervention_cost + treatment_cost_per_tree * (S + E + I)
        else:
            intervention_cost_total = 0
        
        tree_loss_cost = D * tree_value
        total_cost = surveillance_cost + intervention_cost_total + tree_loss_cost
        
        total_costs[sr_idx, rep] = total_cost
        mortality[sr_idx, rep] = D / N_trees * 100

mean_costs = np.mean(total_costs, axis=1) / 1000
std_costs = np.std(total_costs, axis=1) / 1000

optimal_idx = np.argmin(mean_costs[1:]) + 1

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# Hatch patterns for different sampling rates
hatches = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']

# ============================================================================
# Panel A: Total cost (HATCH DIFFERENTIATION)
# ============================================================================

ax1 = axes[0]

bars = ax1.bar(sampling_rates * 100, mean_costs, 
               color='0.7', edgecolor='black', linewidth=1)

# Apply hatches
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Highlight optimal with darker fill
bars[optimal_idx].set_color('0.4')

ax1.errorbar(sampling_rates * 100, mean_costs, yerr=std_costs,
             fmt='none', ecolor='black', capsize=4, capthick=1.5, alpha=0.6)

ax1.set_xlabel('Sampling rate (% trees/month)', fontsize=11)
ax1.set_ylabel('Total cost (€1000s)', fontsize=11)
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax1.grid(True, alpha=0.2, linestyle=':', axis='y')
ax1.set_xscale('log')

# ============================================================================
# Panel B: Cost breakdown (pie chart)
# ============================================================================

ax2 = axes[1]

surveillance_cost_opt = sampling_cost * sampling_rates[optimal_idx] * N_trees * time_horizon / 1000
intervention_cost_opt = (intervention_cost + treatment_cost_per_tree * 600) / 1000
tree_loss_cost_opt = np.mean(mortality[optimal_idx]) / 100 * N_trees * tree_value / 1000

cost_components = [surveillance_cost_opt, intervention_cost_opt, tree_loss_cost_opt]
labels = ['Surveillance', 'Intervention', 'Tree loss']
# Color-blind friendly colors for cost breakdown
colors_pie = ['#3498db', '#f39c12', '#e74c3c']  # Blue, Orange, Red
hatches_pie = ['///', '|||', '...']

wedges, texts, autotexts = ax2.pie(cost_components, labels=labels, autopct='%1.1f%%', 
                                     colors=colors_pie, startangle=90)

# Apply hatches to wedges for additional differentiation
for wedge, hatch in zip(wedges, hatches_pie):
    wedge.set_hatch(hatch)

ax2.set_title(f'B: Breakdown at {sampling_rates[optimal_idx]*100:.1f}%',
              fontsize=11, fontweight='bold')

# ============================================================================
# Panel C: ROI (BLACK SOLID + FILLED CIRCLES)
# ============================================================================

ax3 = axes[2]

baseline_cost = mean_costs[0]
savings = baseline_cost - mean_costs
investment = sampling_rates * N_trees * time_horizon * sampling_cost / 1000

roi = np.zeros(len(sampling_rates))
for i in range(1, len(sampling_rates)):
    if investment[i] > 0:
        roi[i] = savings[i] / investment[i]

ax3.plot(sampling_rates[1:] * 100, roi[1:], 'o-', 
         linewidth=2, markersize=6, color='black',
         markerfacecolor='black')

# Highlight optimal
ax3.plot([sampling_rates[optimal_idx] * 100], [roi[optimal_idx]], 
         'o', markersize=10, color='black', 
         markerfacecolor='white', markeredgewidth=2, zorder=10)

ax3.set_xlabel('Sampling rate (% trees/month)', fontsize=11)
ax3.set_ylabel('ROI (€ saved per € invested)', fontsize=11)
ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax3.grid(True, alpha=0.2, linestyle=':')
ax3.set_xscale('log')

# ============================================================================
# Panel D: Sensitivity analysis
# ============================================================================

ax4 = axes[3]

tree_values = np.array([300, 400, 500, 600, 700])
optimal_rates = np.zeros(len(tree_values))

for tv_idx, tv in enumerate(tree_values):
    costs_tv = np.zeros(len(sampling_rates))
    for sr_idx, sr in enumerate(sampling_rates):
        surv_cost = sampling_cost * sr * N_trees * time_horizon
        tree_loss = np.mean(mortality[sr_idx]) / 100 * N_trees * tv
        if sr > 0:
            interv_cost = intervention_cost + treatment_cost_per_tree * 600
        else:
            interv_cost = 0
        costs_tv[sr_idx] = (surv_cost + interv_cost + tree_loss) / 1000
    
    optimal_rates[tv_idx] = sampling_rates[np.argmin(costs_tv[1:]) + 1] * 100

ax4.plot(tree_values, optimal_rates, 'o-', 
         linewidth=2, markersize=6, color='black',
         markerfacecolor='black')

ax4.set_xlabel('Tree value (€)', fontsize=11)
ax4.set_ylabel('Optimal sampling rate (%)', fontsize=11)
ax4.set_title('D', fontsize=12, fontweight='bold', loc='left')
ax4.grid(True, alpha=0.2, linestyle=':')

plt.tight_layout()
plt.savefig('FigS2_Economic_Analysis_REVISED_BW.png', dpi=300, bbox_inches='tight')
plt.savefig('FigS2_Economic_Analysis_REVISED_BW.pdf', bbox_inches='tight')
print("Supplementary Figure S2 saved (B/W optimized)")
print(f"  Optimal: {sampling_rates[optimal_idx]*100:.1f}%, Savings: €{savings[optimal_idx]:.0f}k")
print("  Strategy: Timing-based detection approach")
plt.close()