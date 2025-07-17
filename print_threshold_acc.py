import pandas as pd

# Load data
df = pd.read_csv("qwen_evaluation_result_explanation_top5_complete.csv")

# Baseline top-1 accuracy
original_accuracy = df["is_top1_correct"].mean()
print(f"Original Top-1 Accuracy: {original_accuracy:.2%}\n")

# Finer-grained thresholds from 0.0 to 1.0 (step size = 0.05)
thresholds = [round(i * 0.05, 2) for i in range(21)]  # 0.0, 0.05, ..., 1.0

for threshold in thresholds:

    def corrected_prediction(row):
        if row["conf1"] < threshold and pd.notna(row["qwen_class_id"]):
            return row["qwen_class_id"]
        else:
            return row["pred1"]

    df["corrected_prediction"] = df.apply(corrected_prediction, axis=1)
    df["is_correct_after_qwen"] = df["corrected_prediction"] == df["true_class_id"]

    corrected_accuracy = df["is_correct_after_qwen"].mean()

    # Rows where Qwen responded
    valid_qwen_mask = df["qwen_class_id"].notna()
    accuracy_on_valid_qwen = df[valid_qwen_mask]["is_correct_after_qwen"].mean()

    # % of samples corrected (Qwen responded + low confidence)
    corrected_sample_mask = (df["conf1"] < threshold) & valid_qwen_mask
    corrected_sample_percent = corrected_sample_mask.sum() * 100 / 8000

    # % of all samples that are low confidence
    low_conf_percent = (df["conf1"] < threshold).sum() * 100 / 8000

    print(
        f"Threshold < {threshold:.2f} → Accuracy: {corrected_accuracy:.2%} "
        f"(Δ = {corrected_accuracy - original_accuracy:+.2%}), "
        f"Corrected Samples: {corrected_sample_percent:.2f}%, "
        f"Conf Samples: {low_conf_percent:.2f}%, "
        f"Accuracy on valid Qwen Responses: {accuracy_on_valid_qwen:.2%}"
    )
