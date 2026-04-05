import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Define approximate coordinates for major cities (simplified map)
cities = {
    'Stockholm': (8.5, 6.5),
    'Uppsala': (8.0, 5.8),
    'Järna': (7.8, 6.0),
    'Katrineholm': (6.5, 5.0),
    'Åby': (5.0, 3.5),
    'Lund': (4.2, 2.8),
    'Malmö': (4.0, 2.5),
    'Skavstaby': (8.2, 6.2),
    'Rosersberg': (8.0, 6.0),
    'Myrbacken': (7.9, 5.9),
    'Blackvreten': (7.8, 5.8),
    'Baggetorp': (6.0, 4.5)
}

# Create network graph
G = nx.Graph()

# Add nodes
for city, pos in cities.items():
    G.add_node(city, pos=pos)

# Define routes with colors and labels
routes = {
    'Stockholm-Uppsala': {
        'path': ['Stockholm', 'Skavstaby', 'Rosersberg', 'Myrbacken', 'Blackvreten', 'Uppsala'],
        'color': 'blue',
        'width': 3,
        'label': 'Commuter Route\n(Low Risk)'
    },
    'Route 1 (Stockholm-Malmö)': {
        'path': ['Stockholm', 'Järna', 'Åby', 'Lund', 'Malmö'],
        'color': 'green', 
        'width': 2,
        'label': 'Route 1\n(Medium Risk)'
    },
    'Route 2 (Stockholm-Malmö)': {
        'path': ['Stockholm', 'Järna', 'Katrineholm', 'Baggetorp', 'Lund', 'Malmö'],
        'color': 'red',
        'width': 2,
        'label': 'Route 2\n(High Risk)'
    },
    'Route 3 (Stockholm-Malmö)': {
        'path': ['Stockholm', 'Järna', 'Katrineholm', 'Åby', 'Lund', 'Malmö'],
        'color': 'orange',
        'width': 2,
        'label': 'Route 3\n(Low Risk)'
    }
}

# Plot the routes
for route_name, route_info in routes.items():
    path = route_info['path']
    color = route_info['color']
    width = route_info['width']
    
    # Add edges to graph
    for i in range(len(path)-1):
        G.add_edge(path[i], path[i+1])
    
    # Plot the route
    positions = [cities[city] for city in path]
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    ax.plot(x_coords, y_coords, color=color, linewidth=width, alpha=0.8, marker='o')
    
    # Add route label at midpoint
    mid_idx = len(path) // 2
    if mid_idx < len(path):
        label_city = path[mid_idx]
        x, y = cities[label_city]
        # Offset the label slightly
        ax.text(x+0.1, y+0.1, route_info['label'], fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                color=color, weight='bold')

# Plot all cities
for city, (x, y) in cities.items():
    # Size nodes based on importance
    if city in ['Stockholm', 'Malmö', 'Uppsala']:
        size = 200
        color = 'darkblue'
    else:
        size = 80
        color = 'lightblue'
    
    ax.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors='black', linewidth=1)
    ax.text(x+0.05, y+0.05, city, fontsize=9, ha='left', va='bottom', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

# Add risk analysis highlights
risk_analysis = [
    ("Route 2: Highest CVaR₉₉\n157 min worst-case", cities['Katrineholm'], 'red'),
    ("Route 3: Most Reliable\n115 min worst-case", cities['Åby'], 'orange'),
    ("Uppsala Route:\nStable & Resilient", cities['Uppsala'], 'blue'),
    ("Critical Junctions:\nRoute divergence points", cities['Järna'], 'purple')
]

for text, pos, color in risk_analysis:
    ax.annotate(text, xy=pos, xytext=(pos[0]+0.5, pos[1]+0.5),
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9),
                fontsize=8, ha='left')

# Customize the plot
ax.set_title('Swedish Railway Network Analysis\nRSP-CVaR Optimization Study', 
             fontsize=16, fontweight='bold', pad=20)

# Create custom legend
legend_elements = [
    plt.Line2D([0], [0], color='blue', lw=3, label='Stockholm-Uppsala (Low Risk)'),
    plt.Line2D([0], [0], color='green', lw=2, label='Route 1 (Medium Risk)'),
    plt.Line2D([0], [0], color='red', lw=2, label='Route 2 (High Risk)'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Route 3 (Low Risk)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
               markersize=8, label='Major Cities'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=6, label='Intermediate Stations')
]

ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

# Add performance metrics box
performance_text = (
    "Key Performance Metrics:\n"
    "• Route 2 CVaR₉₉: 157 min (37% worse than Route 3)\n"
    "• Route 3 CVaR₉₉: 115 min (Most reliable)\n"
    "• All routes: Similar mean delays (3.1-3.8 min)\n"
    "• Risk profiles differ significantly"
)

ax.text(0.02, 0.98, performance_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=1", facecolor='lightgreen', alpha=0.8),
        verticalalignment='top')

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add background color for better presentation
ax.set_facecolor('#f5f5f5')

plt.tight_layout()
plt.show()

# Print route statistics for reference
print("ROUTE PERFORMANCE SUMMARY")
print("=" * 50)
print("Stockholm-Uppsala Route:")
print("  - Type: Commuter route")
print("  - Performance: Stable, low delays (<1.5 min)")
print("  - Risk: Low")
print()

print("Stockholm-Malmö Routes:")
print("  Route 1: Medium risk, balanced performance")
print("  Route 2: HIGH RISK - Worst CVaR (157 min)")
print("  Route 3: LOW RISK - Best reliability (115 min CVaR)")
print()
print("Key Insight: Similar average delays but dramatically different risk profiles!")