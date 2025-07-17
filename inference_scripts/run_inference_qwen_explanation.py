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
import tiktoken  # Token counter

load_dotenv()

# === CONFIGURATION ===
IMAGE_DIR = "../dataset/semi-aves/test"
TOP3_CSV = "../prediction_analysis_stage1_top5.csv"
OUTPUT_CSV = "../qwen_output/qwen_top3_explanations_stage1.csv"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
THROTTLE_SEC = 0.5
TIMEOUT_SEC = 60

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# Load prediction data
df = pd.read_csv(TOP3_CSV)

# Fallback for scientific names
df["pred1_sci"] = df.get("pred1_sci", df["pred1_name"])
df["pred2_sci"] = df.get("pred2_sci", df["pred2_name"])
df["pred3_sci"] = df.get("pred3_sci", df["pred3_name"])

# Already processed images
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done_paths = set(done_df["filename"].tolist())
else:
    done_paths = set()

# === Tokenizer Setup ===
encoding = tiktoken.encoding_for_model("gpt-4")  # or fallback to cl100k_base

def count_tokens(text):
    return len(encoding.encode(text))

# === Utility Functions ===
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Prompt 0
def build_explanation_prompt(row):
    return (
        "You are a helpful assistant that identifies bird species from images.\n\n"
        "Step 1: Carefully examine the bird in the image and note its distinguishing features "
        "(such as color, shape, size, beak, wings, or habitat).\n\n"
        "Step 2: Compare these features to the following three species:\n\n"
        f"1. {row['pred1_name']} ({row['pred1_sci']})\n"
        f"2. {row['pred2_name']} ({row['pred2_sci']})\n"
        f"3. {row['pred3_name']} ({row['pred3_sci']})\n\n"
        "Step 3: Choose the most likely species and explain why it is a better match than the other two.\n\n"
        "Respond in the following format:\n\n"
        "Most Likely: [1, 2, or 3]\n"
        "Explanation: [your explanation here]"
    )

# Prompt 1
# def build_explanation_prompt(row):
#     return (
#         "You are a helpful assistant that identifies bird species from images.\n\n"
#         "Step 1: Examine the bird in the image and describe its visible features "
#         "(such as color, size, beak shape, markings, and habitat).\n\n"
#         "Step 2: Compare those features carefully with each of the following species:\n"
#         f"1. {row['pred1_name']} ({row['pred1_sci']})\n"
#         f"2. {row['pred2_name']} ({row['pred2_sci']})\n"
#         f"3. {row['pred3_name']} ({row['pred3_sci']})\n\n"
#         "Step 3: Reason through which species is the best match, explaining why each alternative is more or less likely.\n\n"
#         "Step 4: Based on your reasoning above, decide which species is the most likely match.\n\n"
#         "Respond in the following format:\n\n"
#         "Reasoning: [detailed comparison and analysis here]\n"
#         "Most Likely: [1, 2, or 3]"
#     )

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
    if row["filename"] in done_paths:
        continue

    image_path = os.path.join(IMAGE_DIR, os.path.basename(row["filename"]))

    if not os.path.exists(image_path):
        print(f"[{idx}] Skipping missing image: {image_path}")
        continue

    try:
        Image.open(image_path).verify()
    except Exception as e:
        print(f"[{idx}] Corrupted image file: {row['filename']}")
        with open("qwen_error_log.txt", "a") as logf:
            logf.write(f"{row['filename']}: corrupted image\n")
        continue

    try:
        # print(f"[{idx}] {datetime.now().strftime('%H:%M:%S')} - Processing {row['image_path']}")
        base64_image = encode_image(image_path)
        prompt = build_explanation_prompt(row)

        prompt_tokens = count_tokens(prompt)
        # print(f"[{idx}]Prompt tokens: {prompt_tokens}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_qwen, prompt, base64_image)
            response = future.result(timeout=TIMEOUT_SEC)

        answer = response.choices[0].message.content.strip()
        response_tokens = count_tokens(answer)
        total_tokens = prompt_tokens + response_tokens
        total_tokens_used += total_tokens

        # print(f"[{idx}] Tokens used: {total_tokens} | Total so far: {total_tokens_used}")

        pd.DataFrame([{
            "image_path": row["filename"],
            "prompt": prompt,
            "response": answer
        }]).to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

        time.sleep(THROTTLE_SEC)

    except concurrent.futures.TimeoutError:
        print(f"[{idx}] Timeout while calling Qwen for {row['filename']}")
        with open("qwen_error_log.txt", "a") as logf:
            logf.write(f"{row['filename']}: timeout\n")
        continue

    except Exception as e:
        print(f"[{idx}] Error with image {row['filename']}: {e}")
        with open("qwen_error_log.txt", "a") as logf:
            logf.write(f"{row['filename']}: {str(e)}\n")
        continue
