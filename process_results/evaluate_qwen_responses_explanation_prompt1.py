import pandas as pd
import re

# === File paths ===
PRED_PATH = "../prediction_analysis.csv"
QWEN_PATH = "../qwen_output/qwen_top3_explanations_prompt1.csv"

# === Load data ===
pred_df = pd.read_csv(PRED_PATH)
qwen_df = pd.read_csv(QWEN_PATH)

# === Merge on image_path ===
df = pred_df.merge(qwen_df, on="image_path", how="inner")

# === Extract Qwen's choice and explanation ===
def extract_choice_and_expl(response):
    if not isinstance(response, str):
        return None, None

    response_lower = response.lower()

    # Case 1: Refusal (e.g., "none of the options" or "none of the provided options")
    if "none of the options" in response_lower or "none of the provided options" in response_lower:
        expl = response.split("Most Likely")[0].strip() if "Most Likely" in response else response.strip()
        return None, expl

    # Case 2: Standard case — parse choice (1, 2, or 3)
    match = re.search(r"most likely\s*[:\-]?\s*(\d)", response, re.IGNORECASE)
    choice = int(match.group(1)) if match else None

    # Get explanation (before 'Most Likely' if present)
    expl = response.strip()
    if match:
        expl = response[:match.start()].strip()

    return choice, expl

# Apply parser
df[["qwen_choice", "qwen_explanation"]] = df["response"].apply(
    lambda x: pd.Series(extract_choice_and_expl(x))
)

# === Map Qwen's choice to class ID ===
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

# Save invalid responses
invalid_df = df[df["qwen_class_id"].isna()]
invalid_df.to_csv("../error_logs/qwen_explanation_prompt1_invalid_responses.csv", index=False)
print(f"Invalid responses saved: {len(invalid_df)}")

# Determine correctness
df["is_correct"] = df["qwen_class_id"] == df["true_class_id"]

# Save final results
df.to_csv("../post_processed_csv/qwen_evaluation_result_explanation_prompt1.csv", index=False)

# Print stats
total = len(df)
valid = df["qwen_class_id"].notna().sum()
correct = df["is_correct"].sum()
accuracy = correct / valid if valid else 0

print(f"Total images: {total}")
print(f"Qwen gave valid answers: {valid}")
print(f"Correct predictions: {correct}")
print(f"Qwen Accuracy (on valid): {accuracy:.2%}")
