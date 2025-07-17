import os
import base64
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
from PIL import Image
import tiktoken

load_dotenv()

# === CONFIGURATION ===
IMAGE_DIR = "../dataset/semi-aves/test"
TOP5_CSV = "../prediction_analysis_fsft_top5.csv"
OUTPUT_CSV = "../qwen_output/qwen_top5_explanations_fsft.csv"
ERROR_FILE = "../error_logs/qwen_error_log_top5_explanation_fsft.txt"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
THROTTLE_SEC = 0.5
TIMEOUT_SEC = 60

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Load error paths ===
with open(ERROR_FILE, "r") as f:
    error_paths = [line.strip().split(":")[0] for line in f if line.strip()]
error_paths = set(os.path.basename(p) for p in error_paths)

# === Load prediction data ===
df = pd.read_csv(TOP5_CSV)
df = df.rename(columns={"filename": "image_path"})  # if necessary

# Filter rows that match error_paths
df["basename"] = df["image_path"].apply(os.path.basename)
df = df[df["basename"].isin(error_paths)].copy()

# Fallback for scientific names
for i in range(1, 6):
    df[f'pred{i}_sci'] = df.get(f'pred{i}_sci', df[f'pred{i}_name'])

# Load and deduplicate output file
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done_df = done_df[~done_df["image_path"].apply(os.path.basename).isin(error_paths)]  # remove duplicates
    done_df.to_csv(OUTPUT_CSV, index=False)
    done_paths = set(done_df["image_path"].tolist())
else:
    done_paths = set()

# === Tokenizer Setup ===
encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    return len(encoding.encode(text))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def build_explanation_prompt(row):
    return (
        "You are a helpful assistant that identifies bird species from images.\n\n"
        "Step 1: Carefully examine the bird in the image and note its distinguishing features "
        "(such as color, shape, size, beak, wings, or habitat).\n\n"
        "Step 2: Compare these features to the following five species:\n\n"
        f"1. {row['pred1_name']} ({row['pred1_sci']})\n"
        f"2. {row['pred2_name']} ({row['pred2_sci']})\n"
        f"3. {row['pred3_name']} ({row['pred3_sci']})\n"
        f"4. {row['pred4_name']} ({row['pred4_sci']})\n"
        f"5. {row['pred5_name']} ({row['pred5_sci']})\n\n"
        "Step 3: Choose the most likely species and explain why it is a better match than the other four.\n\n"
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

# === Run Inference ===
total_tokens_used = 0

for idx, (i, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
    if row["image_path"] in done_paths:
        continue

    image_path = os.path.join(IMAGE_DIR, os.path.basename(row["image_path"]))
    if not os.path.exists(image_path):
        print(f"[{idx}] Missing: {image_path}")
        continue

    try:
        Image.open(image_path).verify()
    except Exception as e:
        print(f"[{idx}] Corrupted: {row['image_path']}")
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

        pd.DataFrame([{
            "image_path": row["image_path"],
            "prompt": prompt,
            "response": answer
        }]).to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

        time.sleep(THROTTLE_SEC)

    except concurrent.futures.TimeoutError:
        print(f"[{idx}] Timeout: {row['image_path']}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: timeout\n")
        continue

    except Exception as e:
        print(f"[{idx}] Error: {row['image_path']}: {e}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: {str(e)}\n")
        continue
