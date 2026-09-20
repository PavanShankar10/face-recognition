"""
encode_faces.py - Feature Extraction Stage (Stage 1)
----------------------------------------------------
Walks through the dataset directory (dataset/<person_name>/<images>),
detects faces using a pretrained deep neural network, computes 128-dimensional
face embeddings, and serializes them to 'encodings.pickle'.

Usage:
    python encode_faces.py
"""

import os
import sys
import pickle
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Import helper functions
from utils import load_image_as_rgb, extract_face_encodings, FACE_REC_AVAILABLE, DEEPFACE_AVAILABLE


# Supported image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_DIR = Path("dataset")
ENCODINGS_FILE = Path("encodings.pickle")


def main():
    print("=" * 65)
    print("  STAGE 1: FACE EMBEDDING EXTRACTION (Deep Feature Extractor)")
    print("=" * 65)

    # 1. Validate that dataset directory exists
    if not DATASET_DIR.exists():
        print(f"[ERROR] The dataset directory '{DATASET_DIR}' does not exist.")
        print("Creating 'dataset/' directory now...")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Find all person subdirectories
    subdirs = [p for p in DATASET_DIR.iterdir() if p.is_dir()]

    if not subdirs:
        print("\n[INFO] No person folders found in 'dataset/'.")
        print("\n--> HOW TO POPULATE YOUR DATASET:")
        print("    Create a subfolder inside 'dataset/' for each person, like this:")
        print("      dataset/")
        print("      |-- Alice/")
        print("      |   |-- photo1.jpg")
        print("      |   |-- photo2.jpg")
        print("      |   `-- photo3.jpg")
        print("      `-- Bob/")
        print("          |-- img1.png")
        print("          `-- img2.jpg")
        print("\n    Recommendation: Add at least 3-5 clear face photos per person.")
        print("    After adding photos, run: python encode_faces.py\n")
        sys.exit(0)

    # 3. Check for library availability
    if not FACE_REC_AVAILABLE and not DEEPFACE_AVAILABLE:
        print("\n[ERROR] Neither 'face_recognition' nor 'deepface' is installed.")
        print("Please install requirements using:")
        print("    pip install -r requirements.txt\n")
        sys.exit(1)

    known_encodings = []
    known_names = []
    person_counts = {}
    skipped_files = []
    total_images_scanned = 0

    print(f"\n[INFO] Scanning dataset across {len(subdirs)} identity folder(s)...")

    for person_dir in sorted(subdirs):
        person_name = person_dir.name
        image_files = [f for f in person_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]

        if not image_files:
            print(f"[WARNING] Folder '{person_name}' is empty (no .jpg, .jpeg, or .png files found).")
            continue

        print(f"\n--> Processing '{person_name}' ({len(image_files)} images)...")
        person_encoded_count = 0

        for img_path in image_files:
            total_images_scanned += 1
            try:
                # Load image as standard RGB
                rgb_img = load_image_as_rgb(str(img_path))

                # Detect face and compute 128-d embedding
                boxes, encodings = extract_face_encodings(rgb_img)

                if len(encodings) == 0:
                    print(f"    [SKIPPED] No face detected in: {img_path.name}")
                    skipped_files.append((str(img_path), "No face detected"))
                    continue

                # In training dataset, we use the primary detected face
                encoding = encodings[0]
                known_encodings.append(encoding)
                known_names.append(person_name)
                person_encoded_count += 1
                print(f"    [+] Encoded: {img_path.name}")

            except Exception as e:
                print(f"    [ERROR] Failed processing {img_path.name}: {e}")
                skipped_files.append((str(img_path), str(e)))

        person_counts[person_name] = person_encoded_count

    # 4. Check if we gathered any valid embeddings
    if not known_encodings:
        print("\n" + "!" * 65)
        print("[ERROR] No face encodings could be extracted from your dataset.")
        print("Please ensure your photos contain clear, unobstructed human faces.")
        print("!" * 65 + "\n")
        sys.exit(1)

    # 5. Serialize embeddings to disk
    print(f"\n[INFO] Saving {len(known_encodings)} face encodings to '{ENCODINGS_FILE}'...")
    data = {
        "encodings": np.array(known_encodings),
        "names": known_names
    }
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    # 6. Print detailed summary
    print("\n" + "=" * 65)
    print("                    ENCODING SUMMARY")
    print("=" * 65)
    print(f"  Total Images Scanned    : {total_images_scanned}")
    print(f"  Total Encodings Saved   : {len(known_encodings)}")
    print(f"  Total Distinct People   : {len(person_counts)}")
    print("-" * 65)
    print("  Per-Person Breakdown:")
    for person, count in person_counts.items():
        print(f"    * {person:<25}: {count} image(s) encoded")
    
    if skipped_files:
        print("-" * 65)
        print(f"  [WARNING] Skipped {len(skipped_files)} file(s) due to missing face or read errors.")
    print("=" * 65)
    print(f"[SUCCESS] Encodings saved to '{ENCODINGS_FILE.resolve()}'")
    print("Next step: Run 'python train_model.py' to train the classifier.\n")


if __name__ == "__main__":
    main()
