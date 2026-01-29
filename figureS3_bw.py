"""
REVISED Supplementary Figure S1: Network Robustness
BLACK-WHITE OPTIMIZED:
- Nodes: uniform grey, differentiate by SIZE
- Time series: Different line styles
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

np.random.seed(789)

network_configs = {
    'Random': {'type': 'random', 'n_nodes': 200, 'edge_prob': 0.05},
    'Small-world': {'type': 'small_world', 'n_nodes': 200, 'k_neighbors': 6, 'rewire_prob': 0.1},
    'Hub': {'type': 'hub', 'n_nodes': 200, 'hub_fraction': 0.05, 'hub_connectivity': 0.4},
    'Grid': {'type': 'grid', 'n_nodes': 196}
}

def create_network(config):
    n = config['n_nodes']
    
    if config['type'] == 'random':
        G = nx.erdos_renyi_graph(n, config['edge_prob'])
    elif config['type'] == 'small_world':
        G = nx.watts_strogatz_graph(n, config['k_neighbors'], config['rewire_prob'])
    elif config['type'] == 'hub':
        G = nx.Graph()
        G.add_nodes_from(range(n))
        n_hubs = int(n * config['hub_fraction'])
        hubs = list(range(n_hubs))
        regular = list(range(n_hubs, n))
        
        for i in range(len(hubs)):
            for j in range(i+1, len(hubs)):
                if np.random.random() < config['hub_connectivity']:
                    G.add_edge(hubs[i], hubs[j])
        
        for hub in hubs:
            n_connections = int(len(regular) * config['hub_connectivity'])
            targets = np.random.choice(regular, n_connections, replace=False)
            for target in targets:
                G.add_edge(hub, target)
        
        for i in range(len(regular)):
            for j in range(i+1, min(i+4, len(regular))):
                if np.random.random() < 0.3:
                    G.add_edge(regular[i], regular[j])
    elif config['type'] == 'grid':
        side = int(np.sqrt(n))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
        num_long_range = int(n * 0.02)
        nodes = list(G.nodes())
        for _ in range(num_long_range):
            u, v = np.random.choice(nodes, 2, replace=False)
            G.add_edge(u, v)
    
    return G

def simulate_cryptic_invasion(G, time_steps=60):
    n = G.number_of_nodes()
    state = ['S'] * n
    
    initial_infected = np.random.choice(n, min(3, n), replace=False)
    for node in initial_infected:
        state[node] = 'E'
    
    infection_timeline = []
    
    lambda_beetle = 0.02
    beta_infection = 0.75
    sigma_latent = 0.05
    mu_disease = 0.03
    
    for t in range(time_steps):
        new_state = state.copy()
        infected_nodes = [i for i, s in enumerate(state) if s in ['E', 'I']]
        
        for node in range(n):
            if state[node] == 'S':
                neighbors = list(G.neighbors(node))
                infected_neighbors = [s for s in neighbors if state[s] in ['E', 'I']]
                
                if len(infected_neighbors) > 0:
                    reinforcement = 1 + 2.0 * len(infected_neighbors) / max(1, len(neighbors))
                    if np.random.random() < lambda_beetle * reinforcement:
                        if np.random.random() < beta_infection:
                            new_state[node] = 'E'
                
                if len(infected_nodes) > 0 and np.random.random() < 0.001:
                    if np.random.random() < beta_infection:
                        new_state[node] = 'E'
            
            elif state[node] == 'E':
                if np.random.random() < sigma_latent:
                    new_state[node] = 'I'
            
            elif state[node] == 'I':
                if np.random.random() < mu_disease:
                    new_state[node] = 'D'
        
        state = new_state
        infected_count = sum(1 for s in state if s in ['E', 'I'])
        infection_timeline.append(infected_count / n * 100)
    
    return infection_timeline, state

# Run simulations
results = {}
for name, config in network_configs.items():
    print(f"Simulating {name}...")
    G = create_network(config)
    infection, final_state = simulate_cryptic_invasion(G)
    
    results[name] = {
        'graph': G,
        'infection': infection,
        'final_state': final_state,
        'config': config
    }

# Create figure
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Node state colors (color-blind friendly palette) + sizes
state_colors = {
    'S': '#2ecc71',  # Green (healthy)
    'E': '#f39c12',  # Orange (latent)
    'I': '#e74c3c',  # Red (diseased)
    'D': '#95a5a6'   # Grey (dead)
}

state_sizes = {
    'S': 20,   # Small (healthy)
    'E': 35,   # Medium (latent)
    'I': 50,   # Large (diseased)
    'D': 70    # Largest (dead) - increased for better visibility
}

# ============================================================================
# Top row: Network visualizations (COLOR-CODED BY STATE + SIZE)
# ============================================================================

for idx, (name, data) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, idx])
    
    G = data['graph']
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Color-code nodes by state, size also varies (double encoding)
    node_colors = [state_colors[state] for state in data['final_state']]
    node_sizes = [state_sizes[state] for state in data['final_state']]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=node_sizes, alpha=0.8, ax=ax,
                          edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)
    
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.axis('off')

# ============================================================================
# Middle row: Infection dynamics (LINESTYLE DIFFERENTIATION)
# ============================================================================

ax_middle = fig.add_subplot(gs[1, :])

time_months = np.arange(60)

# Different LINE STYLES + COLORS for each network (color-blind friendly)
network_styles = {
    'Random': {'linestyle': '-', 'color': '#3498db', 'linewidth': 2},      # Blue solid
    'Small-world': {'linestyle': '--', 'color': '#e74c3c', 'linewidth': 2}, # Red dashed
    'Hub': {'linestyle': '-', 'color': '#2ecc71', 'linewidth': 2.5},        # Green solid (thicker)
    'Grid': {'linestyle': ':', 'color': '#9b59b6', 'linewidth': 2}          # Purple dotted
}

for name, data in results.items():
    style = network_styles[name]
    ax_middle.plot(time_months, data['infection'], 
                  linewidth=style['linewidth'], label=name, 
                  color=style['color'], 
                  linestyle=style['linestyle'])

ax_middle.set_xlabel('Time (months)', fontsize=11)
ax_middle.set_ylabel('Infected trees (%)', fontsize=11)
ax_middle.set_title('Network structure robustness', fontsize=12, fontweight='bold')
ax_middle.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax_middle.grid(True, alpha=0.2, linestyle=':')
ax_middle.set_ylim(0, 100)

# ============================================================================
# Bottom row: Network metrics
# ============================================================================

metrics_names = ['Degree', 'Clustering', 'Path length', 'Final inf. (%)']
metrics_data = []

for name, data in results.items():
    G = data['graph']
    avg_degree = np.mean([d for n, d in G.degree()])
    clustering = nx.average_clustering(G)
    
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        avg_path = nx.average_shortest_path_length(subG)
    
    final_inf = data['infection'][-1]
    metrics_data.append([avg_degree, clustering, avg_path, final_inf])

metrics_data = np.array(metrics_data).T

for i, metric_name in enumerate(metrics_names):
    ax = fig.add_subplot(gs[2, i])
    
    x = np.arange(len(network_configs))
    bars = ax.bar(x, metrics_data[i], color='0.7', alpha=0.7, 
                  edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, metrics_data[i]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(list(network_configs.keys()), 
                       rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.2, linestyle=':', axis='y')

plt.tight_layout()
plt.savefig('FigS3_Network_Robustness_REVISED_BW.png', dpi=300, bbox_inches='tight')
plt.savefig('FigS3_Network_Robustness_REVISED_BW.pdf', bbox_inches='tight')
print("Supplementary Figure S3 saved (B/W optimized)")
final_infections = metrics_data[3]
print(f"  Mean: {np.mean(final_infections):.1f}%, CV: {np.std(final_infections)/np.mean(final_infections)*100:.1f}%")
plt.close()