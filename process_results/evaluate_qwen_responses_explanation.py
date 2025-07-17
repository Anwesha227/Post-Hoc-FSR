import pandas as pd

# === File paths ===
PRED_PATH = "../prediction_analysis_stage1_top5.csv"
QWEN_PATH = "../qwen_output/qwen_top3_explanations_stage1.csv"

# === Load data ===
pred_df = pd.read_csv(PRED_PATH)
qwen_df = pd.read_csv(QWEN_PATH)

# === Merge on image_path ===
df = pred_df.merge(qwen_df, on="filename", how="inner")

# === Extract Qwen's choice (1, 2, or 3) from the 'response' field ===
def extract_qwen_choice_and_explanation(response):
    if not isinstance(response, str):
        return None, None

    choice = None
    explanation = None
    lines = response.strip().splitlines()

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if line_lower.startswith("most likely"):
            for char in line:
                if char in {"1", "2", "3"}:
                    choice = int(char)
                    break
        elif line_lower.startswith("explanation:"):
            # Get everything after "Explanation:"
            explanation = line.partition("Explanation:")[2].strip()
            # If explanation spans multiple lines, join those
            if i + 1 < len(lines) and not lines[i + 1].lower().startswith("most likely"):
                explanation += "\n" + "\n".join(lines[i+1:])

    return choice, explanation

df[["qwen_choice", "qwen_explanation"]] = df["response"].apply(
    lambda x: pd.Series(extract_qwen_choice_and_explanation(x))
)

# === Map Qwen's choice to predicted class ID ===
def map_choice_to_class(row):
    if row["qwen_choice"] == 1:
        return row["pred1"]
    elif row["qwen_choice"] == 2:
        return row["pred2"]
    elif row["qwen_choice"] == 3:
        return row["pred3"]
    else:
        return None

df["qwen_class_id"] = df.apply(map_choice_to_class, axis=1)

# Save invalid responses (no valid qwen_choice or class mapping)
invalid_df = df[df["qwen_class_id"].isna()]

# Optional: Save to CSV for inspection
invalid_df.to_csv("../qwen_invalid_responses.csv", index=False)

print(f"Invalid responses saved to qwen_invalid_responses.csv: {len(invalid_df)} rows")

# === Determine correctness ===
df["is_correct"] = df["qwen_class_id"] == df["true_class_id"]

# === Save results ===
df.to_csv("../qwen_evaluation_result_explanation_stage1.csv", index=False)

# === Print stats ===
total = len(df)
valid = df["qwen_class_id"].notna().sum()
correct = df["is_correct"].sum()
accuracy = correct / valid if valid else 0

print(f"Total images: {total}")
print(f"Qwen gave valid answers: {valid}")
print(f"Correct predictions: {correct}")
print(f"Qwen Accuracy (on valid): {accuracy:.2%}")
