import pandas as pd
import os

# === Choose version ===
VERSION = "complete"  # options: "sci" or "complete"

# === File paths ===
PRED_PATH = "../prediction_analysis_top5.csv"
QWEN_PATH = f"../qwen_output/qwen_top5_explanations_{VERSION}.csv"
OUTPUT_CSV = f"../post_processed_csv/qwen_evaluation_result_explanation_top5_{VERSION}.csv"
INVALID_CSV = f"../error_logs/qwen_invalid_responses_top5_{VERSION}.csv"

# === Load data ===
pred_df = pd.read_csv(PRED_PATH)
qwen_df = pd.read_csv(QWEN_PATH)

# === Normalize filenames if needed
pred_df = pred_df.rename(columns={"filename": "image_path"})

# === Merge on image_path ===
df = pred_df.merge(qwen_df, on="image_path", how="inner")

# === Extract Qwen's choice and explanation ===
def extract_qwen_choice_and_explanation(response):
    if not isinstance(response, str):
        return None, None

    choice = None
    explanation = None
    lines = response.strip().splitlines()

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        if "most likely" in line_lower:
            for token in line:
                if token in {"1", "2", "3", "4", "5"}:
                    choice = int(token)
                    break

        if line_lower.startswith("explanation:"):
            explanation = line.partition("Explanation:")[2].strip()
            if i + 1 < len(lines):
                extra_lines = lines[i + 1:]
                explanation += "\n" + "\n".join(extra_lines).strip()
            break

    return choice, explanation

df[["qwen_choice", "qwen_explanation"]] = df["response"].apply(
    lambda x: pd.Series(extract_qwen_choice_and_explanation(x))
)

# === Map Qwen's choice to class ID ===
def map_choice_to_class(row):
    choice = row["qwen_choice"]
    if choice == 1:
        return row.get("pred1")
    elif choice == 2:
        return row.get("pred2")
    elif choice == 3:
        return row.get("pred3")
    elif choice == 4:
        return row.get("pred4")
    elif choice == 5:
        return row.get("pred5")
    return None

df["qwen_class_id"] = df.apply(map_choice_to_class, axis=1)

# === Determine correctness ===
df["is_correct"] = df["qwen_class_id"] == df["true_class_id"]

# === Save invalid responses ===
invalid_df = df[df["qwen_class_id"].isna()]
invalid_df.to_csv(INVALID_CSV, index=False)
print(f"Invalid responses saved to {INVALID_CSV}: {len(invalid_df)} rows")

# === Save evaluation results ===
df.to_csv(OUTPUT_CSV, index=False)

# === Print summary ===
total = len(df)
valid = df["qwen_class_id"].notna().sum()
correct = df["is_correct"].sum()
accuracy = correct / valid if valid else 0

print(f"Version: {VERSION}")
print(f"Total images: {total}")
print(f"Qwen gave valid answers: {valid}")
print(f"Correct predictions: {correct}")
print(f"Qwen Accuracy (on valid): {accuracy:.2%}")
