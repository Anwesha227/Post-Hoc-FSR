import os
import base64
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# === Load environment variables ===
load_dotenv()

# === CONFIGURATION ===
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
QUERY_IMAGE_DIR = "../dataset/semi-aves/test"
PREGNERATED_REF_DIR = "../dataset/semi-aves/pregenerated_references_6shot"
CSV_PATH = "../prediction_analysis_top5.csv"
METADATA_JSON = "../dataset/semi-aves/semi-aves_metrics-LAION400M-updated.json"
OUTPUT_CSV = "../qwen_output/qwen_multimodal_top5_6shot_results.csv"
SKIPPED_LOG = "../error_logs/qwen_multimodal_top5_6shot_skipped.txt"
TIMEOUT_SEC = 120

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Load class ID mapping ===
with open(METADATA_JSON, "r") as f:
    id2info = json.load(f)

sci_to_classid = {
    v["most_common_name"].lower().strip(): int(k)
    for k, v in id2info.items()
}

# === Image Encoding Functions ===
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_stitched_image_from_class(class_id):
    stitched_path = os.path.join(PREGNERATED_REF_DIR, f"{class_id}.jpg")
    if not os.path.exists(stitched_path):
        raise FileNotFoundError(f"Stitched image missing: {stitched_path}")
    with open(stitched_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_multimodal_prompt(query_image, candidates):
    messages = [
        {"type": "text", "text":
         "You are a helpful assistant that identifies birds using a test image and visual references.\n\n"
         "Step 1: Look at the first image — this is the bird we want to identify.\n"
         "Step 2: Compare it to each of the 5 candidate species. Each candidate includes the species name and a stitched image showing six different individuals of that species.\n"
         "Step 3: Select the most likely match and explain why.\n\n"
         "Respond in this format:\n"
         "Most Likely: [1, 2, 3, 4, or 5]\n"
         "Reasoning: [your explanation here]"
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}}
    ]
    for idx, (name, img) in enumerate(candidates, 1):
        messages.append({"type": "text", "text": f"Candidate {idx}: {name}"})
        messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    return messages

# === Load existing output to avoid duplication ===
already_done = set()
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    already_done.update(done_df["image_path"].tolist())

# === Read skipped entries ===
with open(SKIPPED_LOG, "r") as f:
    skipped_files = [line.split(" - ")[0].strip() for line in f if line.strip()]

# === Load input data and filter to skipped but not-yet-processed entries ===
df = pd.read_csv(CSV_PATH)
df = df[df["filename"].isin(skipped_files) & ~df["filename"].isin(already_done)]

# Add fallback for pred_sci columns
for i in range(1, 6):
    df[f"pred{i}_sci"] = df.get(f"pred{i}_sci", df[f"pred{i}_name"])

skipped = []

# === Run inference for remaining skipped entries ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image_path = os.path.join(QUERY_IMAGE_DIR, os.path.basename(row["filename"]))
        if not os.path.exists(image_path):
            skipped.append((row["filename"], "Missing test image"))
            continue

        query_img_base64 = encode_image(image_path)

        candidates = []
        for i in range(1, 6):
            sci_name = row[f"pred{i}_sci"].lower().strip()
            class_id = sci_to_classid.get(sci_name)
            if class_id is None:
                raise KeyError(f"Missing species: {sci_name}")
            ref_img = get_stitched_image_from_class(class_id)
            candidates.append((row[f"pred{i}_sci"], ref_img))

        messages = build_multimodal_prompt(query_img_base64, candidates)

        # Qwen inference
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": messages}],
                    temperature=0.4,
                    max_tokens=900,
                )
                response = future.result(timeout=TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            skipped.append((row["filename"], "Timeout during Qwen call"))
            continue

        result_row = {
            "image_path": row["filename"],
            "pred1_sci": row["pred1_sci"],
            "pred2_sci": row["pred2_sci"],
            "pred3_sci": row["pred3_sci"],
            "pred4_sci": row["pred4_sci"],
            "pred5_sci": row["pred5_sci"],
            "response": response.choices[0].message.content.strip()
        }
        pd.DataFrame([result_row]).to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

    except Exception as e:
        skipped.append((row["filename"], str(e)))
        continue

# === Save updated skipped log ===
os.makedirs(os.path.dirname(SKIPPED_LOG), exist_ok=True)
with open(SKIPPED_LOG, "w") as f:
    for path, reason in skipped:
        f.write(f"{path} - {reason}\n")

print(f"\nRerun complete. Appended {len(df) - len(skipped)} new entries to {OUTPUT_CSV}. Skipped again: {len(skipped)}")
