import pandas as pd

# === File paths ===
PRED_PATH = "../prediction_analysis.csv"
QWEN_PATH = "../qwen_output/qwen_image_only_results.csv"

# === Load data ===
pred_df = pd.read_csv(PRED_PATH)
qwen_df = pd.read_csv(QWEN_PATH)

# === Merge on image_path ===
df = pred_df.merge(qwen_df, on="image_path", how="inner")

# === Extract Qwen's choice (1, 2, or 3) from the 'response' field ===
def extract_choice_and_reasoning(response):
    if not isinstance(response, str):
        return None, None

    choice = None
    reasoning = None
    lines = response.strip().splitlines()

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if line_lower.startswith("most likely"):
            for char in line:
                if char in {"1", "2", "3"}:
                    choice = int(char)
                    break
        elif line_lower.startswith("reasoning:"):
            reasoning = line.partition("Reasoning:")[2].strip()
            if i + 1 < len(lines) and not lines[i + 1].lower().startswith("most likely"):
                reasoning += "\n" + "\n".join(lines[i+1:])

    return choice, reasoning

df[["qwen_choice", "qwen_reasoning"]] = df["response"].apply(
    lambda x: pd.Series(extract_choice_and_reasoning(x))
)

# === Map Qwen's choice to class ID ===
def map_choice(row):
    if row["qwen_choice"] == 1:
        return row["pred1"]
    elif row["qwen_choice"] == 2:
        return row["pred2"]
    elif row["qwen_choice"] == 3:
        return row["pred3"]
    else:
        return None

df["qwen_class_id"] = df.apply(map_choice, axis=1)

# === Save invalid rows ===
invalid_df = df[df["qwen_class_id"].isna()]
invalid_df.to_csv("../qwen_output/qwen_image_only_invalid.csv", index=False)
print(f"Invalid responses saved: {len(invalid_df)} rows")

# === Evaluate accuracy ===
df["is_correct"] = df["qwen_class_id"] == df["true_class_id"]

# === Save final evaluation ===
df.to_csv("../qwen_output/qwen_evaluation_result_image_only.csv", index=False)

# === Print stats ===
total = len(df)
valid = df["qwen_class_id"].notna().sum()
correct = df["is_correct"].sum()
accuracy = correct / valid if valid else 0

print(f"Total images: {total}")
print(f"Qwen gave valid answers: {valid}")
print(f"Correct predictions: {correct}")
print(f"Qwen Accuracy (on valid): {accuracy:.2%}")
