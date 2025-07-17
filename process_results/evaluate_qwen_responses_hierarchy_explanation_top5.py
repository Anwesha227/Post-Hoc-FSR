import pandas as pd
import re
import os

# === File paths ===
PRED_PATH = "../prediction_analysis_top5_with_families.csv"
QWEN_PATH = "../qwen_output/qwen_hierarchical_explanations_top5.csv"
OUTPUT_CSV = "../post_processed_csv/qwen_evaluation_result_explanation_top5_hierarchical.csv"
INVALID_CSV = "../error_logs/qwen_invalid_responses_top5_hierarchical.csv"

# === Load data ===
pred_df = pd.read_csv(PRED_PATH)
qwen_df = pd.read_csv(QWEN_PATH)

# === Normalize filenames for matching ===
pred_df["image_name"] = pred_df["filename"].apply(lambda x: os.path.basename(str(x)).lower().strip())
qwen_df["image_name"] = qwen_df["image_path"].str.lower().str.strip()

# === Merge on normalized image name
df = pred_df.merge(qwen_df, on="image_name", how="inner")

# === Normalize species name
def normalize(name):
    if pd.isna(name):
        return ""
    return name.strip().lower()

df["response"] = df["response"].astype(str)
df["normalized_response_species"] = df["response"].str.extract(r"most likely:\s*(.*)", flags=re.IGNORECASE)[0].apply(normalize)

# === Normalize prediction names
for i in range(1, 6):
    df[f"pred{i}_name_norm"] = df[f"pred{i}_name"].apply(normalize)

# === Match Qwen's species response to prediction class ID
def find_class_id(row):
    resp = row["normalized_response_species"]
    for i in range(1, 6):
        if resp == row.get(f"pred{i}_name_norm"):
            return row.get(f"pred{i}")
    return None

df["qwen_class_id"] = df.apply(find_class_id, axis=1)

# === Extract explanation
def extract_explanation(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lower().strip().startswith("reasoning:") or line.lower().strip().startswith("explanation:"):
            return line.partition(":")[2].strip() + "\n" + "\n".join(lines[i+1:]).strip()
    return ""

df["qwen_explanation"] = df["response"].apply(extract_explanation)

# === Determine correctness
df["is_correct"] = df["qwen_class_id"] == df["true_class_id"]

# === Save invalid responses
invalid_df = df[df["qwen_class_id"].isna()]
invalid_df.to_csv(INVALID_CSV, index=False)
print(f"Invalid responses saved to {INVALID_CSV}: {len(invalid_df)} rows")

# === Save results
df.to_csv(OUTPUT_CSV, index=False)

# === Summary
total = len(df)
valid = df["qwen_class_id"].notna().sum()
correct = df["is_correct"].sum()
accuracy = correct / valid if valid else 0

print(f"Total images: {total}")
print(f"Qwen gave valid answers: {valid}")
print(f"Correct predictions: {correct}")
print(f"Qwen Accuracy (on valid): {accuracy:.2%}")
