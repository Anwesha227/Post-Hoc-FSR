import os
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

REFERENCE_IMAGE_DIR = "human_selected_4shot"
FEWSHOT_FILE = "fewshot8_seed1.txt"
OUTPUT_DIR = "pregenerated_references_8shot"
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

    # Step 2: Load the 4 human-selected images
    human_images = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(human_images) < 4:
        raise ValueError(f"Not enough human-selected images in class {class_id}")
    
    selected_human = random.sample(human_images, 4)
    human_imgs = [Image.open(os.path.join(class_path, img)).convert("RGB") for img in selected_human]

    # Step 3: Load 4 additional images from fewshot file (excluding human-selected)
    fewshot_paths = class_to_paths[class_id]
    extra_candidates = [p for p in fewshot_paths if os.path.basename(p) not in selected_human]

    if len(extra_candidates) < 4:
        raise ValueError(f"Not enough extra fewshot images for class {class_id}")

    selected_extra = random.sample(extra_candidates, 4)
    extra_imgs = [Image.open(p).convert("RGB") for p in selected_extra]
    imgs = human_imgs + extra_imgs

    # === Try 4-4 split and pick most balanced ===
    best_split = None
    min_width_diff = float("inf")

    # All possible balanced 4-4 splits (first 4 vs last 4)
    candidate_splits = [
        (imgs[:4], imgs[4:])
    ]

    for row1_imgs, row2_imgs in candidate_splits:
        h1 = min(img.height for img in row1_imgs)
        h2 = min(img.height for img in row2_imgs)

        w1 = sum(int(img.width * h1 / img.height) for img in row1_imgs)
        w2 = sum(int(img.width * h2 / img.height) for img in row2_imgs)

        diff = abs(w1 - w2)
        if diff < min_width_diff:
            min_width_diff = diff
            best_split = (row1_imgs, h1, w1, row2_imgs, h2, w2)

    row1_imgs, h1, w1, row2_imgs, h2, w2 = best_split

    row1_resized = [img.resize((int(img.width * h1 / img.height), h1)) for img in row1_imgs]
    row2_resized = [img.resize((int(img.width * h2 / img.height), h2)) for img in row2_imgs]

    stitched_width = max(w1, w2)
    stitched_height = h1 + h2

    stitched = Image.new("RGB", (stitched_width, stitched_height))

    x_offset = (stitched_width - w1) // 2
    for img in row1_resized:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width

    x_offset = (stitched_width - w2) // 2
    for img in row2_resized:
        stitched.paste(img, (x_offset, h1))
        x_offset += img.width

    stitched.save(output_path, format="JPEG")


# === Run for all classes ===
skipped = []

for class_id in tqdm(range(NUM_CLASSES)):
    try:
        create_stitched_reference(class_id)
    except Exception as e:
        skipped.append((class_id, str(e)))

print(f"\nDone generating tiles.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} classes:")
    for cid, reason in skipped:
        print(f"  Class {cid}: {reason}")
