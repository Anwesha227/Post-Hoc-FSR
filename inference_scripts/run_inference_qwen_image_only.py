import os
import base64
import random
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

load_dotenv()

# === CONFIGURATION ===
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
QUERY_IMAGE_DIR = "../dataset/semi-aves/test"
REFERENCE_IMAGE_DIR = "../dataset/semi-aves/human_selected_4shot"
CSV_PATH = "../prediction_analysis.csv"
METADATA_JSON = "../dataset/semi-aves/semi-aves_metrics-LAION400M-updated.json"
OUTPUT_CSV = "../qwen_output/qwen_image_only_results.csv"
SKIPPED_LOG = "../error_logs/qwen_image_only_skipped.txt"
TIMEOUT_SEC = 60

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Load common name → class ID mapping ===
with open(METADATA_JSON, "r") as f:
    id2info = json.load(f)

sci_to_classid = {
    v["most_common_name"].lower().strip(): int(k)
    for k, v in id2info.items()
}

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_two_images_from_class(class_id):
    class_path = os.path.join(REFERENCE_IMAGE_DIR, str(class_id))
    all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(all_images) < 2:
        raise ValueError(f"Not enough images in class {class_id}")
    selected = random.sample(all_images, 2)
    return [encode_image(os.path.join(class_path, f)) for f in selected]

def build_image_only_prompt(query_image, candidate_refs):
    messages = [
        {"type": "text", "text":
         "You are a helpful assistant that identifies birds based only on images.\n\n"
         "Step 1: Look at the first image — this is the bird we want to identify.\n"
         "Step 2: Compare it to each of the following 3 candidate groups. Each group contains example images of a bird species.\n"
         "Step 3: Pick the group that best matches and explain why.\n\n"
         "Respond in this format:\n"
         "Most Likely: [1, 2, or 3]\n"
         "Reasoning: [your explanation here]"
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}},
    ]
    for idx, group in enumerate(candidate_refs, 1):
        messages.append({"type": "text", "text": f"Candidate {idx}:"})
        for img in group:
            messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    return messages

# === Prepare dataframe ===
df = pd.read_csv(CSV_PATH)
df["pred1_sci"] = df.get("pred1_sci", df["pred1_name"])
df["pred2_sci"] = df.get("pred2_sci", df["pred2_name"])
df["pred3_sci"] = df.get("pred3_sci", df["pred3_name"])

skipped = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image_path = os.path.join(QUERY_IMAGE_DIR, os.path.basename(row["image_path"]))
        if not os.path.exists(image_path):
            skipped.append((row["image_path"], "Missing test image"))
            continue

        query_img_base64 = encode_image(image_path)

        # Lookup class IDs
        cand1_class = sci_to_classid[row["pred1_sci"].lower().strip()]
        cand2_class = sci_to_classid[row["pred2_sci"].lower().strip()]
        cand3_class = sci_to_classid[row["pred3_sci"].lower().strip()]

        cand1_refs = get_two_images_from_class(cand1_class)
        cand2_refs = get_two_images_from_class(cand2_class)
        cand3_refs = get_two_images_from_class(cand3_class)

        messages = build_image_only_prompt(query_img_base64, [cand1_refs, cand2_refs, cand3_refs])

        # === Qwen call with timeout ===
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
            skipped.append((row["image_path"], "Timeout during Qwen call"))
            continue

        # === Save result immediately
        pd.DataFrame([{
            "image_path": row["image_path"],
            "pred1_sci": row["pred1_sci"],
            "pred2_sci": row["pred2_sci"],
            "pred3_sci": row["pred3_sci"],
            "response": response.choices[0].message.content.strip()
        }]).to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

    except Exception as e:
        skipped.append((row["image_path"], str(e)))
        continue

# === Save skipped log ===
with open(SKIPPED_LOG, "w") as f:
    for path, reason in skipped:
        f.write(f"{path} - {reason}\n")

print(f"\n✅ Done. Appended results to {OUTPUT_CSV}. Skipped {len(skipped)} cases.")
