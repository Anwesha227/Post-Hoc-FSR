import os
import base64
import time
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
from PIL import Image
import tiktoken  # Token counter

load_dotenv()

# === CONFIGURATION ===
IMAGE_DIR = "../dataset/semi-aves/test"
TOP5_CSV = "../prediction_analysis_top5.csv"
OUTPUT_CSV = "../qwen_output/qwen_top5_explanations_complete.csv"
ERROR_FILE = "../error_logs/qwen_error_log_top5_explanation_complete.txt"
TAXONOMY_JSON = "../dataset/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
THROTTLE_SEC = 0.5
TIMEOUT_SEC = 60

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Load Taxonomy Metadata ===
with open(TAXONOMY_JSON, "r") as f:
    taxonomy = json.load(f)

classid_to_metadata = {
    int(cid): {
        "sci": v.get("name", ""),
        "common": v.get("most_common_name", ""),
        "genus": v.get("genus", ""),
        "family": v.get("family", "")
    }
    for cid, v in taxonomy.items()
}

# === Load Prediction Data ===
df = pd.read_csv(TOP5_CSV)
df = df.rename(columns={"filename": "image_path"})

# Add enriched fields to DataFrame
for k in range(1, 6):
    pred_col = f"pred{k}"
    if df[pred_col].dtype != int:
        df[pred_col] = df[pred_col].astype(int)

    df[f"pred{k}_sci"] = df[pred_col].map(lambda cid: classid_to_metadata.get(cid, {}).get("sci", ""))
    df[f"pred{k}_common"] = df[pred_col].map(lambda cid: classid_to_metadata.get(cid, {}).get("common", ""))
    df[f"pred{k}_genus"] = df[pred_col].map(lambda cid: classid_to_metadata.get(cid, {}).get("genus", ""))
    df[f"pred{k}_family"] = df[pred_col].map(lambda cid: classid_to_metadata.get(cid, {}).get("family", ""))

# === Load Error Log ===
if os.path.exists(ERROR_FILE):
    with open(ERROR_FILE, "r") as f:
        error_paths = set([line.strip().split(":")[0] for line in f if ":" in line])
else:
    error_paths = set()

# === Filter rows in error log ===
df = df[df["image_path"].apply(lambda x: x in error_paths)]

# === Tokenizer Setup ===
encoding = tiktoken.encoding_for_model("gpt-4")
def count_tokens(text):
    return len(encoding.encode(text))

# === Utility Functions ===
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def format_species_full(sci, common, genus, family):
    return f"{sci}, also known as {common}, belongs to the genus {genus}, family {family}"

def build_explanation_prompt(row):
    return (
        "You are a helpful assistant that identifies bird species from images.\n\n"
        "Step 1: Carefully examine the bird in the image and note its distinguishing features "
        "(such as color, shape, size, beak, wings, or habitat).\n\n"
        "Step 2: Compare these features to the following five species:\n\n"
        f"1. {format_species_full(row['pred1_sci'], row['pred1_common'], row['pred1_genus'], row['pred1_family'])}\n"
        f"2. {format_species_full(row['pred2_sci'], row['pred2_common'], row['pred2_genus'], row['pred2_family'])}\n"
        f"3. {format_species_full(row['pred3_sci'], row['pred3_common'], row['pred3_genus'], row['pred3_family'])}\n"
        f"4. {format_species_full(row['pred4_sci'], row['pred4_common'], row['pred4_genus'], row['pred4_family'])}\n"
        f"5. {format_species_full(row['pred5_sci'], row['pred5_common'], row['pred5_genus'], row['pred5_family'])}\n\n"
        "Step 3: Choose the most likely species and explain why it is a better match than the other four.\n\n"
        "Respond in the following format:\n\n"
        "Most Likely: [1, 2, 3, 4, or 5]\n"
        "Explanation: [your explanation here]"
    )

def call_qwen(prompt, base64_image):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }},
                ],
            }
        ],
        temperature=0.4,
        max_tokens=900,
    )

# === Inference Loop ===
total_tokens_used = 0

for idx, (i, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
    image_path = os.path.join(IMAGE_DIR, os.path.basename(row["image_path"]))

    if not os.path.exists(image_path):
        print(f"[{idx}] Skipping missing image: {image_path}")
        continue

    try:
        Image.open(image_path).verify()
    except Exception:
        print(f"[{idx}] Corrupted image file: {row['image_path']}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: corrupted image\n")
        continue

    try:
        base64_image = encode_image(image_path)
        prompt = build_explanation_prompt(row)
        prompt_tokens = count_tokens(prompt)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_qwen, prompt, base64_image)
            response = future.result(timeout=TIMEOUT_SEC)

        answer = response.choices[0].message.content.strip()
        response_tokens = count_tokens(answer)
        total_tokens_used += prompt_tokens + response_tokens

        # === Replace old entry for image_path if exists ===
        if os.path.exists(OUTPUT_CSV):
            df_existing = pd.read_csv(OUTPUT_CSV)
            df_existing = df_existing[df_existing["image_path"] != row["image_path"]]
            df_existing.to_csv(OUTPUT_CSV, index=False)

        # === Append new result ===
        new_row = pd.DataFrame([{
            "image_path": row["image_path"],
            "prompt": prompt,
            "response": answer
        }])
        new_row.to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

        time.sleep(THROTTLE_SEC)

    except concurrent.futures.TimeoutError:
        print(f"[{idx}] Timeout while calling Qwen for {row['image_path']}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: timeout\n")
        continue

    except Exception as e:
        print(f"[{idx}] Error with image {row['image_path']}: {e}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: {str(e)}\n")
        continue
