# qwen_positional_tests.py
import os
import random
import json
import base64
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict

# === Load .env from the parent directory ===
load_dotenv(dotenv_path="../.env")

# === CONFIGURATION ===
TEST_PLAN_PATH = "test_plan.csv"
IMAGE_ROOT = "../dataset/semi-aves/test"
LABEL_FILE = "../dataset/semi-aves/test.txt"
METADATA_JSON = "../dataset/semi-aves/semi-aves_metrics-LAION400M-updated.json"
OUTPUT_CSV = "qwen_positional_test_outputs.csv"
N_RUNS_PER_CASE = 5
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Parse class ID to image paths mapping ===
class_to_images = defaultdict(list)
with open(LABEL_FILE, 'r') as f:
    for line in f:
        path, class_id, _ = line.strip().split()
        class_to_images[class_id].append(os.path.join("../dataset/semi-aves", path))

# === Optional: Load class ID to name mapping if needed ===
with open(METADATA_JSON, 'r') as f:
    class_metadata = json.load(f)


def encode_image(image_path):
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")  # Convert to remove alpha channel
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def build_prompt(row, run_images):
    if row["type"] == "positional":
        if row["grouped"]:
            return f"You are shown {int(row['group'])} groups of images. Each group contains equal number of images. Describe the {int(row['target_index'])} image from the {int(row['group'])} group."
        else:
            return f"You are shown {row['num_images']} images. Describe the {ordinal(int(row['target_index']))} image."

    elif row["type"] == "comparison":
        if row["grouped"]:
            return f"You are shown {int(row['group'])} groups of images. Compare the groups and explain how they differ."
        else:
            return f"You are shown {row['num_images']} images. Compare them and describe key visual differences."
    return ""

def ordinal(n):
    return "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def load_class_based_images(n, grouped=False, group_count=1):
    selected_images = []
    available_classes = list(class_to_images.keys())

    if grouped:
        assert n % group_count == 0, "Total images must be divisible by group count"
        per_group = n // group_count
        chosen_classes = random.sample(available_classes, group_count)
        for cls in chosen_classes:
            imgs = random.sample(class_to_images[cls], per_group)
            selected_images.extend(imgs)
    else:
        if n == 3:  # Special case for [A, A, B]
            cls_a, cls_b = random.sample(available_classes, 2)
            selected_images.extend(random.sample(class_to_images[cls_a], 2))
            selected_images.extend(random.sample(class_to_images[cls_b], 1))
        else:
            chosen_classes = random.sample(available_classes, n)
            for cls in chosen_classes:
                selected_images.append(random.choice(class_to_images[cls]))

    return selected_images

def run_test_case(row):
    results = []
    for _ in range(N_RUNS_PER_CASE):
        image_paths = load_class_based_images(
            int(row["num_images"]),
            grouped=row["grouped"],
            group_count=int(row["group"]) if row["grouped"] else 1
        )
        images_encoded = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"}}
            for p in image_paths
        ]
        prompt = build_prompt(row, image_paths)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": images_encoded + [{"type": "text", "text": prompt}]}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[ERROR] {str(e)}"

        results.append({
            "test_id": row["test_id"],
            "prompt": prompt,
            "image_paths": ", ".join(image_paths),
            "response": answer
        })
    return results

if __name__ == "__main__":
    df = pd.read_csv(TEST_PLAN_PATH)
    all_outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_outputs = run_test_case(row)
        all_outputs.extend(row_outputs)

    df_out = pd.DataFrame(all_outputs)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")
