import os
import json
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
from tqdm import tqdm

# === Load prediction CSV and taxonomy metadata ===
df = pd.read_csv("../prediction_analysis_top5.csv")
with open("../dataset/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json", "r") as f:
    taxonomy = json.load(f)

# === Create output directory ===
output_dir = "taxonomy_graphs_json"
os.makedirs(output_dir, exist_ok=True)

def build_taxonomy_graph(row):
    G = nx.DiGraph()
    G.add_node("Aves", level="class", type="class")

    top5_ids = [row["pred1"], row["pred2"], row["pred3"], row["pred4"], row["pred5"]]
    top5_confs = [row["conf1"], row["conf2"], row["conf3"], row["conf4"], row["conf5"]]
    true_id = row["true_class_id"]

    genus_conf_sum = defaultdict(float)
    family_conf_sum = defaultdict(float)

    # === Build hierarchy data for top-5 and possibly true ===
    hierarchy_data = []
    for class_id, conf in zip(top5_ids, top5_confs):
        entry = taxonomy[str(class_id)]
        genus = entry["name"].split()[0]
        family = entry["family"]
        genus_conf_sum[genus] += conf
        family_conf_sum[family] += conf
        hierarchy_data.append({
            "class_id": class_id,
            "common_name": entry["most_common_name"],
            "scientific_name": entry["name"],
            "genus": genus,
            "family": family,
            "confidence": conf,
            "is_top1": class_id == row["pred1"],
            "is_true": class_id == true_id
        })

    # === Add true class if not in top-5 ===
    if true_id not in top5_ids:
        entry = taxonomy[str(true_id)]
        genus = entry["name"].split()[0]
        family = entry["family"]
        hierarchy_data.append({
            "class_id": true_id,
            "common_name": entry["most_common_name"],
            "scientific_name": entry["name"],
            "genus": genus,
            "family": family,
            "confidence": None,
            "is_top1": False,
            "is_true": True
        })

    # === Add "Other" node ===
    top5_sum = sum(c for c in top5_confs if c is not None)
    other_conf = max(0.0, 1.0 - top5_sum)
    G.add_node("Other", confidence=other_conf, type="other")
    G.add_edge("Aves", "Other")

    # === Add family/genus/species nodes and edges ===
    for entry in hierarchy_data:
        fam = entry["family"]
        gen = entry["genus"]
        sci = entry["scientific_name"]
        com = entry["common_name"]
        conf = entry["confidence"]

        # Unique node IDs
        fam_id = f"family::{fam}"
        gen_id = f"genus::{gen}"
        species_id = f"species::{sci}"

        # Labels for display
        fam_conf = family_conf_sum.get(fam, 0.0)
        gen_conf = genus_conf_sum.get(gen, 0.0)

        fam_label = f"{fam}\n({fam_conf:.2f})"
        gen_label = f"{gen}\n({gen_conf:.2f})"
        species_label = f"{com}\n{sci}\n({conf:.2f})" if conf is not None else f"{com}\n{sci}"

        # Add nodes
        G.add_node(fam_id, type="family", confidence=fam_conf, label=fam_label)
        G.add_node(gen_id, type="genus", confidence=gen_conf, label=gen_label)
        G.add_node(
            species_id,
            type="species",
            confidence=conf,
            is_top1=entry["is_top1"],
            is_true=entry["is_true"],
            label=species_label
        )

        # Add edges
        G.add_edge("Aves", fam_id)
        G.add_edge(fam_id, gen_id)
        G.add_edge(gen_id, species_id)

    return G

# === Save graphs ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    G = build_taxonomy_graph(row)
    base_filename = os.path.splitext(os.path.basename(row["filename"]))[0]  # e.g., "4045"
    out_path = os.path.join(output_dir, f"test_{base_filename}.json")
    with open(out_path, "w") as f:
        json.dump(json_graph.node_link_data(G), f)

print("All taxonomy graphs saved to:", output_dir)
