import pandas as pd
import os
import json

# === Paths ===
CSV_PATH = "qwen_positional_test_outputs.csv"
TEST_TXT_PATH = "../dataset/semi-aves/test.txt"
METADATA_JSON_PATH = "../dataset/semi-aves/semi-aves_metrics-LAION400M-updated.json"
OUTPUT_CSV_PATH = "qwen_annotation_with_classes.csv"

# === Load test image -> class ID mapping ===
path_to_classid = {}
with open(TEST_TXT_PATH, "r") as f:
    for line in f:
        path, class_id, _ = line.strip().split()
        filename = os.path.basename(path)
        path_to_classid[filename] = class_id

# === Load class ID -> name mapping ===
with open(METADATA_JSON_PATH, "r") as f:
    metadata = json.load(f)
classid_to_name = {k: v["most_common_name"] for k, v in metadata.items()}

# === Load Qwen output CSV ===
df = pd.read_csv(CSV_PATH)

# === Map test_id to test type ===
test_type_map = {
    'A1': 'Positional (3 images)',
    'A2': 'Positional (5 images)',
    'A3': 'Positional (7 images)',
    'A4': 'Positional Grouped (2x3)',
    'A5': 'Positional Grouped (3x2)',
    'B1': 'Comparison (3 similar)',
    'B2': 'Comparison (5 similar)',
    'B3': 'Comparison ([A, A, B])',
    'B4': 'Group Comparison (2 groups)',
    'B5': 'Group Comparison (3 groups)',
}
df['test_type'] = df['test_id'].map(test_type_map)

# === Derive class names for images ===
def get_class_names(image_path_str):
    names = []
    for rel_path in image_path_str.split(','):
        fname = os.path.basename(rel_path.strip())
        cid = path_to_classid.get(fname)
        cname = classid_to_name.get(cid, "Unknown")
        names.append(cname)
    return ", ".join(names)

df['class_names'] = df['image_paths'].apply(get_class_names)
df['Correct?'] = ''
df['Notes'] = ''

# === Save the final annotated CSV ===
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Saved annotated file to: {OUTPUT_CSV_PATH}")
