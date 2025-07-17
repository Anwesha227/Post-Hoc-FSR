import os
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

REFERENCE_IMAGE_DIR = "human_selected_4shot"
FEWSHOT_FILE = "fewshot8_seed1.txt"
OUTPUT_DIR = "pregenerated_references_4shot"
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

    # Load 4 human-selected images
    human_images = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(human_images) < 4:
        raise ValueError(f"Not enough human-selected images in class {class_id}")
    
    selected_human = random.sample(human_images, 4)
    human_imgs = [Image.open(os.path.join(class_path, img)).convert("RGB") for img in selected_human]

    # Load 2 additional images from fewshot file (prefer no overlap)
    fewshot_paths = class_to_paths[class_id]
    extra_candidates = [p for p in fewshot_paths if os.path.basename(p) not in selected_human]

    if len(extra_candidates) < 2:
        print(f"⚠️ Class {class_id}: Only {len(extra_candidates)} unique fewshot candidates. Allowing overlap.")
        extra_candidates = fewshot_paths

    if len(extra_candidates) < 2:
        raise ValueError(f"Not enough fewshot images even with overlap for class {class_id}")

    selected_extra = random.sample(extra_candidates, 2)
    extra_imgs = [Image.open(p).convert("RGB") for p in selected_extra]

    imgs = human_imgs + extra_imgs
    assert len(imgs) == 6

    # Split into 2 rows of 3
    rows = [imgs[:3], imgs[3:]]

    resized_rows = []
    row_heights = []
    row_widths = []

    for row_imgs in rows:
        h = min(img.height for img in row_imgs)
        row_resized = [img.resize((int(img.width * h / img.height), h)) for img in row_imgs]
        w = sum(img.width for img in row_resized)
        resized_rows.append(row_resized)
        row_heights.append(h)
        row_widths.append(w)

    stitched_width = max(row_widths)
    stitched_height = sum(row_heights)
    stitched = Image.new("RGB", (stitched_width, stitched_height))

    y_offset = 0
    for row_imgs, h, w in zip(resized_rows, row_heights, row_widths):
        x_offset = (stitched_width - w) // 2
        for img in row_imgs:
            stitched.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += h

    stitched.save(output_path, format="JPEG")

# === Run for all classes ===
skipped = []

for class_id in tqdm(range(NUM_CLASSES)):
    try:
        create_stitched_reference(class_id)
    except Exception as e:
        skipped.append((class_id, str(e)))

print(f"\n✅ Done generating 6-shot (2×3) tiles.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} classes:")
    for cid, reason in skipped:
        print(f"  Class {cid}: {reason}")
