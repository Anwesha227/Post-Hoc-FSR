import os
import base64
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# === CONFIGURATION ===
IMAGE_DIR = "dataset/semi-aves/test"
TOP3_CSV = "prediction_analysis.csv"  # Input file with top-3 predictions
OUTPUT_CSV = "qwen_top3_choice.csv"   # Output file to append results
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
THROTTLE_SEC = 0.5

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# === Load Top-3 Predictions CSV ===
df = pd.read_csv(TOP3_CSV)

# Fallback: Use common names as scientific if sci not available
df["pred1_sci"] = df.get("pred1_sci", df["pred1_name"])
df["pred2_sci"] = df.get("pred2_sci", df["pred2_name"])
df["pred3_sci"] = df.get("pred3_sci", df["pred3_name"])

# === Utility Functions ===
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def build_prompt(row):
    option1 = f"{row['pred1_name']} ({row['pred1_sci']})"
    option2 = f"{row['pred2_name']} ({row['pred2_sci']})"
    option3 = f"{row['pred3_name']} ({row['pred3_sci']})"
    return (
        "You are a helpful assistant that identifies bird species from images.\n"
        "Look at the image and choose the most likely bird species from the following options:\n\n"
        f"1. {option1}\n"
        f"2. {option2}\n"
        f"3. {option3}\n\n"
        "Respond only with the number corresponding to the most likely option: 1, 2, or 3."
    )

# === Run Inference ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(IMAGE_DIR, os.path.basename(row["image_path"]))

    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_path}")
        continue

    try:
        base64_image = encode_image(image_path)
        prompt = build_prompt(row)

        response = client.chat.completions.create(
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
            temperature=0.2,
            max_tokens=50,
        )

        answer = response.choices[0].message.content.strip()

        # Save this row immediately
        pd.DataFrame([{
            "image_path": row["image_path"],
            "prompt": prompt,
            "response": answer
        }]).to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

        time.sleep(THROTTLE_SEC)

    except Exception as e:
        print(f"Error with image {row['image_path']}: {e}")
