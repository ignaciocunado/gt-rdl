import networkx as nx
import matplotlib.pyplot as plt

# 1) Create (or load) your graph
G = nx.erdos_renyi_graph(n=120, p=0.4, seed=42)  # example: random graph with 80 nodes

# 2) Pick two "seed" nodes by whatever criterion you like.
#    In a real paper these might be the highest‐centrality nodes or nodes from a k‐shell decomposition.
seed_nodes = [0]  # for example purposes

# 3) Find their immediate neighbors
neighbors_of_seeds = set()
for s in seed_nodes:
    neighbors_of_seeds.update(G.neighbors(s))

# 4) Assign each node a color based on its status:
#      - seed nodes  → dark red
#      - neighbors   → light red/pink
#      - everyone else → gray
node_colors = []
for n in G.nodes():
    if n in seed_nodes:
        node_colors.append('#c70000')
    elif n in neighbors_of_seeds:
        node_colors.append('#c40000')
    else:
        node_colors.append('#000000')

# 5) Run a force‐directed layout so that the graph “spreads out” nicely.
pos = nx.kamada_kawai_layout(G) # Fruchterman‐Reingold (spring) layout

# 6) Draw edges first, then nodes on top
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.reshape((-1))
ax[0].axis('off')
ax[1].axis('off')

nx.draw_networkx_edges(G, pos=pos,
                       edge_color='#000000',    # very light gray edges
                       alpha=0.5,               # faint lines so they don’t distract
                       width=1.0,
                       ax=ax[0])

nx.draw_networkx_nodes(G, pos=pos,
                       node_color=node_colors,
                       node_size=[400 if n not in seed_nodes else 400 for n in G.nodes()],
                       linewidths=0.5,
                       edgecolors='#555555',
                       ax=ax[0])

node_colors = []
for n in G.nodes():
    if n in seed_nodes:
        node_colors.append('#c70000')
    else:
        node_colors.append('#f28b82')

nx.draw_networkx_edges(G, pos=pos,
                       edge_color='#999999',    # very light gray edges
                       alpha=0.3,               # faint lines so they don’t distract
                       width=1.0,
                       ax=ax[1])

nx.draw_networkx_nodes(G, pos=pos,
                       node_color=node_colors,
                       node_size=[60 if n not in seed_nodes else 220 for n in G.nodes()],
                       linewidths=0.5,
                       edgecolors='#555555',
                       ax=ax[1])


plt.axis('off')
plt.tight_layout()
plt.savefig("high_quality_network1.pdf", format='pdf')
plt.show()
