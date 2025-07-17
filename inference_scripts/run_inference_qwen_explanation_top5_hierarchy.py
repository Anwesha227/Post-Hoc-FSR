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
from collections import defaultdict

# === Load environment variables ===
load_dotenv()

# === CONFIGURATION ===
IMAGE_DIR = "../dataset/semi-aves/test"
TOP5_CSV = "../prediction_analysis_top5_with_families.csv"
OUTPUT_CSV = "../qwen_output/qwen_hierarchical_explanations_top5.csv"
ERROR_FILE = "../error_logs/qwen_hierarchical_errors_top5.txt"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
THROTTLE_SEC = 0.5
TIMEOUT_SEC = 60

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Load Prediction Data ===
df = pd.read_csv(TOP5_CSV)
df = df.rename(columns={"filename": "image_path"})
df["image_path"] = df["image_path"].apply(os.path.basename)


# === Tokenizer Setup ===
encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    return len(encoding.encode(text))

# === Image Encoding ===
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# === Prompt Builder ===
def build_hierarchical_prompt(row):
    preds = [
        (row["pred1_name"], row["pred1_family"]),
        (row["pred2_name"], row["pred2_family"]),
        (row["pred3_name"], row["pred3_family"]),
        (row["pred4_name"], row["pred4_family"]),
        (row["pred5_name"], row["pred5_family"]),
    ]

    grouped = defaultdict(list)
    for name, family in preds:
        grouped[family].append(name)

    prompt = (
        "You are a helpful assistant that identifies bird species from images.\n\n"
        "Step 1: Carefully examine the bird in the image and note its distinguishing features "
        "(such as color, shape, size, beak, wings, or habitat).\n\n"
        "Step 2: Compare these features to the following candidate species, organized by family:\n\n"
    )

    for i, (family, species_list) in enumerate(grouped.items(), start=1):
        plural = "species" if len(species_list) > 1 else "species"
        prompt += f"Group {i} — Family: {family} has the following {len(species_list)} possible {plural}:\n"
        for species in species_list:
            prompt += f"- {species}\n"
        prompt += "\n"

    prompt += (
        "Step 3: Using the visual characteristics of the bird, identify the most likely species from the list above. "
        "Groupings are provided to help you understand which species are more closely related, but your decision should be based on visual similarity.\n\n"
        "Respond in the following format:\n"
        "Most Likely: [species name]\n"
        "Reasoning: [your explanation here]"
    )

    return prompt

# === API Call ===
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

# === Resume Support ===
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done_paths = set(done_df["image_path"].tolist())
else:
    done_paths = set()

# === Inference Loop ===
total_tokens_used = 0

for idx, (i, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
    if row["image_path"] in done_paths:
        continue

    full_image_path = os.path.join(IMAGE_DIR, row["image_path"])
    if not os.path.exists(full_image_path):
        print(f"[{idx}] Missing: {full_image_path}")
        continue

    try:
        Image.open(full_image_path).verify()
    except Exception:
        print(f"[{idx}] Corrupt image: {row['image_path']}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: corrupted image\n")
        continue

    try:
        base64_image = encode_image(full_image_path)
        prompt = build_hierarchical_prompt(row)
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
        print(f"[{idx}] Error: {row['image_path']} → {e}")
        with open(ERROR_FILE, "a") as logf:
            logf.write(f"{row['image_path']}: {str(e)}\n")
        continue

print(f"\n=== Total Tokens Used: {total_tokens_used} ===")
