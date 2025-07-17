import os
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

REFERENCE_IMAGE_DIR = "human_selected_4shot"
FEWSHOT_FILE = "fewshot16_seed1.txt"  # Must contain up to 16 entries per class
OUTPUT_DIR = "pregenerated_references_16shot"
NUM_CLASSES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Parse fewshot16_seed1.txt ===
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

    # Load 12 extra images from fewshot file (prefer no overlap with human-selected)
    fewshot_paths = class_to_paths[class_id]
    extra_candidates = [p for p in fewshot_paths if os.path.basename(p) not in selected_human]

    # Fallback: allow overlap if < 12 non-overlapping candidates
    if len(extra_candidates) < 12:
        print(f"⚠️ Class {class_id}: Only {len(extra_candidates)} unique fewshot candidates after removing overlap. Falling back to allow overlap.")
        extra_candidates = fewshot_paths  # Use full list, even if overlap

    if len(extra_candidates) < 12:
        raise ValueError(f"Not enough fewshot images even with overlap for class {class_id}")

    selected_extra = random.sample(extra_candidates, 12)
    extra_imgs = [Image.open(p).convert("RGB") for p in selected_extra]

    # Combine total 16 images
    imgs = human_imgs + extra_imgs
    assert len(imgs) == 16

    # Split into 4 rows of 4 images
    rows = [imgs[i * 4:(i + 1) * 4] for i in range(4)]

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

print(f"\n✅ Done generating 16-shot 4x4 tiles.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} classes:")
    for cid, reason in skipped:
        print(f"  Class {cid}: {reason}")
