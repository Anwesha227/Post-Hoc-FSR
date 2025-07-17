import os
import random
from PIL import Image
from tqdm import tqdm

REFERENCE_IMAGE_DIR = "human_selected_4shot"
OUTPUT_DIR = "pregenerated_references_4shot"
NUM_CLASSES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_stitched_reference(class_id):
    class_path = os.path.join(REFERENCE_IMAGE_DIR, str(class_id))
    output_path = os.path.join(OUTPUT_DIR, f"{class_id}.jpg")

    if os.path.exists(output_path):
        return  # Skip if already exists

    # Load and filter image files
    all_images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(all_images) < 4:
        raise ValueError(f"Not enough images in class {class_id}")

    selected = random.sample(all_images, 4)
    imgs = [Image.open(os.path.join(class_path, img)).convert("RGB") for img in selected]

    # === Try all 3 unique 2-2 splits and pick the most balanced one ===
    best_split = None
    min_width_diff = float("inf")

    candidate_splits = [
        ([imgs[0], imgs[1]], [imgs[2], imgs[3]]),
        ([imgs[0], imgs[2]], [imgs[1], imgs[3]]),
        ([imgs[0], imgs[3]], [imgs[1], imgs[2]]),
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

    # Resize images in each row
    row1_resized = [img.resize((int(img.width * h1 / img.height), h1)) for img in row1_imgs]
    row2_resized = [img.resize((int(img.width * h2 / img.height), h2)) for img in row2_imgs]

    stitched_width = max(w1, w2)
    stitched_height = h1 + h2

    stitched = Image.new("RGB", (stitched_width, stitched_height))

    # Paste top row, center-aligned
    x_offset = (stitched_width - w1) // 2
    for img in row1_resized:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width

    # Paste bottom row, center-aligned
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
