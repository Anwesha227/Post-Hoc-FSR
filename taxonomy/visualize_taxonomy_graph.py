import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph

def visualize_taxonomy_graph(json_path, save_path=None, figsize=(12, 8)):
    with open(json_path, "r") as f:
        G = json_graph.node_link_graph(json.load(f))

    node_colors = []
    labels = {}

    for node, data in G.nodes(data=True):
        # === Color logic ===
        if data.get("type") == "other":
            node_colors.append("#D3D3D3")  # light gray
        elif data.get("is_true"):
            node_colors.append("green")
        elif data.get("is_top1"):
            node_colors.append("red")
        else:
            node_colors.append("#A1C9F4")  # blue

        # === Label logic ===
        if data.get("type") == "other":
            label = f"Other ({data['confidence']:.2f})"
        elif data.get("type") in {"family", "genus"} and data.get("confidence") is not None:
            label = f"{node.splitlines()[0]} ({data['confidence']:.2f})"
        elif data.get("type") == "species" and data.get("confidence") is not None:
            label = f"{node.splitlines()[0]} ({data['confidence']:.2f})"
        else:
            label = node
        labels[node] = label

    # === Layout ===
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        print("⚠️ pygraphviz not found — falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)

    # === Plot ===
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        labels=labels,
        arrows=False,
        node_size=3000,
        node_color=node_colors,
        font_size=9
    )
    plt.title("Taxonomic Hierarchy with Confidence")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved graph to: {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    visualize_taxonomy_graph("taxonomy_graphs_json/test_1105.json")
    # Or to save as PNG:
    # visualize_taxonomy_graph("taxonomy_graphs_json/test_4045.json", save_path="taxonomy_graphs_png/test_4045.png")
