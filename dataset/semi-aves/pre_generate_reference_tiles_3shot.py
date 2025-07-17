import os
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

REFERENCE_IMAGE_DIR = "human_selected_4shot"
FEWSHOT_FILE = "fewshot8_seed1.txt"
OUTPUT_DIR = "pregenerated_references_3shot"
NUM_CLASSES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Parse fewshot8_seed1.txt ===
class_to_paths = defaultdict(list)
with open(FEWSHOT_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        rel_path, class_id = parts[0], int(parts[1])
        class_to_paths[class_id].append(rel_path)


def create_stitched_reference(class_id):
    class_str = str(class_id)
    class_path = os.path.join(REFERENCE_IMAGE_DIR, class_str)
    output_path = os.path.join(OUTPUT_DIR, f"{class_str}.jpg")

    if os.path.exists(output_path):
        return  # Skip if already exists

    # Load 3 human-selected images
    human_images = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(human_images) < 3:
        raise ValueError(f"Not enough human-selected images in class {class_id}")

    selected_human = random.sample(human_images, 3)
    imgs = [Image.open(os.path.join(class_path, img)).convert("RGB") for img in selected_human]

    # Resize images to uniform height
    min_height = min(img.height for img in imgs)
    imgs_resized = [img.resize((int(img.width * min_height / img.height), min_height)) for img in imgs]

    # Calculate stitched dimensions
    stitched_width = sum(img.width for img in imgs_resized)
    stitched_height = min_height

    stitched = Image.new("RGB", (stitched_width, stitched_height))

    x_offset = 0
    for img in imgs_resized:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width

    stitched.save(output_path, format="JPEG")


# === Run for all classes ===
skipped = []

for class_id in tqdm(range(NUM_CLASSES)):
    try:
        create_stitched_reference(class_id)
    except Exception as e:
        skipped.append((class_id, str(e)))

print(f"\n✅ Done generating 3-shot (1×3) tiles.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} classes:")
    for cid, reason in skipped:
        print(f"  Class {cid}: {reason}")