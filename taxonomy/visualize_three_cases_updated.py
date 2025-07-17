import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph

# === CONFIG ===
CSV_PATH = "../prediction_analysis_top5.csv"
GRAPH_DIR = "taxonomy_graphs_json"


# === Load prediction data ===
df = pd.read_csv(CSV_PATH)

# === Find 3 cases ===
case1 = df[(df["is_top5_correct"] == True) & (df["is_top1_correct"] == False)].iloc[0]
case2 = df[df["is_top1_correct"] == True].iloc[0]
case3 = df[df["is_top5_correct"] == False].iloc[0]
cases = [case1, case2, case3]
titles = [
    "(1) Top-5 Correct, Top-1 Wrong",
    "(2) Top-1 Correct",
    "(3) True Class Not in Top-5"
]

# === Visualization helper ===
def visualize_taxonomy_graph(json_path, title, figsize=(12, 8)):
    with open(json_path, "r") as f:
        G = json_graph.node_link_graph(json.load(f))

    node_colors = []
    labels = {}
    max_family_node = None
    max_family_conf = -1.0

    # First pass to determine highest-probability family
    for node, data in G.nodes(data=True):
        if data.get("type") == "family" and data.get("confidence", -1) > max_family_conf:
            max_family_conf = data["confidence"]
            max_family_node = node

    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        confidence = data.get("confidence")
        is_true = data.get("is_true", False)
        is_top1 = data.get("is_top1", False)

        # === Node Coloring ===
        if is_true:
            node_colors.append("green")
        elif is_top1:
            node_colors.append("red")
        elif node == max_family_node:
            node_colors.append("orange")  # Highlight highest-probability family
        elif node_type == "other":
            node_colors.append("#D3D3D3")
        else:
            node_colors.append("#A1C9F4")

        # === Label Formatting ===
        if confidence is not None:
            label = f"{node.splitlines()[0]} ({confidence:.2f})"
        else:
            label = node.splitlines()[0] if "\n" in node else node
        labels[node] = label

    # Layout
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        pos = nx.spring_layout(G, seed=42)

    # Plot
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=node_colors,
        node_size=3000,
        arrows=False,
        font_size=9
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Visualize the 3 cases ===
for row, title in zip(cases, titles):
    img_name = os.path.splitext(os.path.basename(row["filename"]))[0]
    graph_path = os.path.join(GRAPH_DIR, f"test_{img_name}.json")
    visualize_taxonomy_graph(graph_path, title)
