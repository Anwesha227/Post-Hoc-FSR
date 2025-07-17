import pandas as pd

# === File paths ===
PRED_PATH = "../prediction_analysis.csv"
QWEN_PATH = "../qwen_top3_choice.csv"

# === Load data ===
pred_df = pd.read_csv(PRED_PATH)
qwen_df = pd.read_csv(QWEN_PATH)

# === Merge on image_path ===
df = pred_df.merge(qwen_df, on="image_path", how="inner")

# === Extract Qwen's choice (1, 2, or 3) ===
def extract_qwen_choice(response):
    if not isinstance(response, str):
        return None
    response = response.strip()
    if response.lower().startswith("none"):
        return None
    for char in response:
        if char in {"1", "2", "3"}:
            return int(char)
    return None

df["qwen_choice"] = df["response"].apply(extract_qwen_choice)

# === Map Qwen's choice to predicted class ===
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

# === Determine correctness ===
df["is_correct"] = df["qwen_class_id"] == df["true_class_id"]

# === Optional: Save merged results ===
df.to_csv("../qwen_evaluation_result_top3.csv", index=False)

# === Print quick stats ===
total = len(df)
valid = df["qwen_class_id"].notna().sum()
correct = df["is_correct"].sum()
accuracy = correct / valid if valid else 0

print(f"Total images: {total}")
print(f"Qwen gave valid answers: {valid}")
print(f"Correct predictions: {correct}")
print(f"Qwen Accuracy (on valid): {accuracy:.2%}")
