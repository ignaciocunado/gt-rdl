import networkx as nx
import matplotlib.pyplot as plt

G = nx.erdos_renyi_graph(n=120, p=0.4, seed=42)

seed_nodes = [0,1]

neighbors_of_seeds = set()
for s in seed_nodes:
    neighbors_of_seeds.update(G.neighbors(s))

node_colors = []
for n in G.nodes():
    if n in seed_nodes:
        node_colors.append('#c70000')
    elif n in neighbors_of_seeds:
        node_colors.append('#c40000')
    else:
        node_colors.append('#000000')

pos = nx.kamada_kawai_layout(G)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.reshape((-1))
ax[0].axis('off')
ax[1].axis('off')

nx.draw_networkx_edges(G, pos=pos,
                       edge_color='#000000',
                       alpha=0.5,
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
                       edge_color='#999999',
                       alpha=0.3,
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
