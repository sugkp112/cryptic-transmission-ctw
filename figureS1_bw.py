"""
REVISED Supplementary Figure S2: Latency Distribution Robustness
BLACK-WHITE OPTIMIZED:
- Exponential: solid
- Gamma: dashed
- Log-normal: dash-dot
- Weibull: thick solid
NOTE: Derived metric is consistent with the CTW framework used in the main text
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

np.random.seed(101112)

# LINE STYLE + COLOR mapping for distributions (color-blind friendly)
latency_models = {
    'Exponential': {
        'type': 'exponential',
        'mean': 20,
        'linestyle': '-',      # solid
        'linewidth': 2,
        'color': '#3498db'     # Blue
    },
    'Gamma': {
        'type': 'gamma',
        'shape': 3,
        'scale': 7,
        'linestyle': '--',     # dashed
        'linewidth': 2,
        'color': '#e74c3c'     # Red
    },
    'Log-normal': {
        'type': 'lognormal',
        'mean': 2.5,
        'sigma': 0.8,
        'linestyle': '-.',     # dash-dot
        'linewidth': 2,
        'color': '#2ecc71'     # Green
    },
    'Weibull': {
        'type': 'weibull',
        'shape': 2.5,
        'scale': 22,
        'linestyle': '-',      # thick solid
        'linewidth': 3,
        'color': '#9b59b6'     # Purple
    }
}

def get_latency_time(model_config, size=1):
    if model_config['type'] == 'exponential':
        return np.random.exponential(model_config['mean'], size)
    elif model_config['type'] == 'gamma':
        return np.random.gamma(model_config['shape'], model_config['scale'], size)
    elif model_config['type'] == 'lognormal':
        return np.random.lognormal(model_config['mean'], model_config['sigma'], size)
    elif model_config['type'] == 'weibull':
        return model_config['scale'] * np.random.weibull(model_config['shape'], size)

def simulate_with_latency_dist(model_config, time_steps=80, n_replicates=50):
    N_trees = 500
    lambda_beetle = 0.025
    beta_infection = 0.75
    mu_disease = 0.03
    alpha_reinforcement = 2.5
    
    all_infections = np.zeros((n_replicates, time_steps))
    all_timing_metric = np.zeros((n_replicates, time_steps))
    
    for rep in range(n_replicates):
        S, I, D = N_trees, 0, 0
        E_individuals = []
        
        for t in range(time_steps):
            reinforcement = 1 + alpha_reinforcement * (len(E_individuals) + I) / N_trees
            beetle_events = np.random.poisson(lambda_beetle * S * reinforcement)
            new_inf = min(S, np.random.binomial(beetle_events, beta_infection))
            
            beetle_visible = beetle_events * 0.15
            
            if new_inf > 0:
                latency_times = get_latency_time(model_config, new_inf)
                E_individuals.extend(latency_times)
                S -= new_inf
            
            E_individuals = [time - 1 for time in E_individuals]
            newly_symptomatic = sum(1 for time in E_individuals if time <= 0)
            E_individuals = [time for time in E_individuals if time > 0]
            I += newly_symptomatic
            
            i_to_d = np.random.binomial(I, mu_disease)
            I -= i_to_d
            D += i_to_d
            
            E_count = len(E_individuals)
            all_infections[rep, t] = (E_count + I) / N_trees * 100
            all_timing_metric[rep, t] = (E_count + I) / (beetle_visible + 0.1)
    
    return all_infections, all_timing_metric

# Run simulations
print("Simulating different latency distributions...")
results = {}
for name, config in latency_models.items():
    print(f"  {name}...")
    infections, timing_metric = simulate_with_latency_dist(config)
    results[name] = {
        'infections': infections,
        'timing_metric': timing_metric,
        'config': config
    }

# Create figure
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

time_months = np.arange(80)

# ============================================================================
# Panel A: Latency distributions
# ============================================================================

ax1 = fig.add_subplot(gs[0, :2])

x = np.linspace(0.1, 60, 500)
for name, config in latency_models.items():
    if config['type'] == 'exponential':
        pdf = stats.expon.pdf(x, scale=config['mean'])
    elif config['type'] == 'gamma':
        pdf = stats.gamma.pdf(x, config['shape'], scale=config['scale'])
    elif config['type'] == 'lognormal':
        pdf = stats.lognorm.pdf(x, config['sigma'], scale=np.exp(config['mean']))
    elif config['type'] == 'weibull':
        pdf = stats.weibull_min.pdf(x, config['shape'], scale=config['scale'])
    
    ax1.plot(x, pdf, 
             linewidth=config['linewidth'], 
             linestyle=config['linestyle'],
             color=config['color'],  # Use color from config
             label=name)

ax1.set_xlabel('Latency period (months)', fontsize=11)
ax1.set_ylabel('Probability density (month⁻¹)', fontsize=11)
ax1.set_title('A: Distribution models', fontsize=11, fontweight='bold', loc='left')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.2, linestyle=':')
ax1.set_xlim(0, 60)

# ============================================================================
# Panel B: Example realizations
# ============================================================================

ax2 = fig.add_subplot(gs[0, 2:])

n_examples = 100
for name, config in latency_models.items():
    latency_samples = get_latency_time(config, n_examples)
    y_positions = np.random.uniform(-0.2, 0.2, n_examples)
    
    ax2.scatter(latency_samples, [list(latency_models.keys()).index(name)] + y_positions,
               alpha=0.4, s=20, color='0.5', marker='.')

ax2.set_yticks(range(len(latency_models)))
ax2.set_yticklabels(latency_models.keys(), fontsize=10)
ax2.set_xlabel('Sampled latency period (months)', fontsize=11)
ax2.set_title('B: Random samples', fontsize=11, fontweight='bold', loc='left')
ax2.grid(True, alpha=0.2, linestyle=':', axis='x')
ax2.set_xlim(0, 60)

# ============================================================================
# Panel C: Infection dynamics
# ============================================================================

ax3 = fig.add_subplot(gs[1, :])

for name, data in results.items():
    config = data['config']
    mean_inf = np.mean(data['infections'], axis=0)
    std_inf = np.std(data['infections'], axis=0)
    
    ax3.plot(time_months, mean_inf, 
            linewidth=config['linewidth'], 
            linestyle=config['linestyle'],
            color=config['color'],  # Use color from config
            label=name)
    ax3.fill_between(time_months, mean_inf - std_inf, mean_inf + std_inf,
                    alpha=0.15, color=config['color'])  # Matching color fill

ax3.set_xlabel('Time (months)', fontsize=11)
ax3.set_ylabel('Infected trees (%)', fontsize=11)
ax3.set_title('C: Infection dynamics', fontsize=11, fontweight='bold', loc='left')
ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax3.grid(True, alpha=0.2, linestyle=':')
ax3.set_ylim(0, 85)

# ============================================================================
# Panel D: Derived metric trajectories
# ============================================================================

ax4 = fig.add_subplot(gs[2, :])

for name, data in results.items():
    config = data['config']
    mean_metric = np.mean(data['timing_metric'], axis=0)
    
    ax4.plot(time_months, mean_metric, 
            linewidth=config['linewidth'], 
            linestyle=config['linestyle'],
            color=config['color'],  # Use color from config
            alpha=0.9)

ax4.axhline(y=5, color='0.6', linestyle=':', linewidth=1.5, alpha=0.6)

ax4.set_xlabel('Time (months)', fontsize=11)
ax4.set_ylabel('Derived pre-detection transmission metric(a.u.)', fontsize=11)
ax4.set_title('D: Derived metric trajectories', fontsize=11, fontweight='bold', loc='left')
ax4.grid(True, alpha=0.2, linestyle=':')
ax4.set_ylim(0, 1500)  # Increased to accommodate initial spikes

plt.tight_layout()
plt.savefig('Fig1_Latency_Robustness_REVISED_BW.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig1_Latency_Robustness_REVISED_BW.pdf', bbox_inches='tight')
print("Figure 1 saved (B/W optimized - linestyle differentiation)")
print("Note: Derived metric is consistent with the CTW framework used in the main text")
plt.close()