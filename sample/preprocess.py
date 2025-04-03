import os
import numpy as np
import cv2
from tqdm import tqdm

# 🔹 Set paths (Windows-compatible)
DATASET_PATH = r"C:\oct project\sample\train"
IMAGE_PATH = os.path.join(DATASET_PATH, "img")
LABEL_PATH = os.path.join(DATASET_PATH, "labels.npy")
OUTPUT_PATH = r"C:\oct project\sample\processed_data"

# 🔹 Image processing parameters
IMAGE_SIZE = (224, 224)  # Resize images to 224x224

def extract_label_from_filename(filename):
    """Extract label from filename. Modify this function based on your naming convention."""
    return filename.split("_")[0]  # Assumes "class1_001.jpg" -> "class1"

def generate_labels(image_files):
    """Generate labels based on filenames if labels.npy is missing."""
    print("⚠ No labels.npy found. Generating labels from filenames...")
    labels = [extract_label_from_filename(img) for img in image_files]

    # Convert labels to integers if they are class names
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in labels])  # Convert to numbers

    print("✅ Generated labels:", label_map)
    return labels

def preprocess_data():
    """Preprocess images and save as NumPy arrays."""
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"❌ Error: Image folder '{IMAGE_PATH}' not found!")

    image_files = sorted(os.listdir(IMAGE_PATH))
    print(f"📂 Found {len(image_files)} images. Processing...")

    # Load or generate labels
    if os.path.exists(LABEL_PATH):
        print("✅ Labels found! Loading...")
        labels = np.load(LABEL_PATH, allow_pickle=True)
    else:
        labels = generate_labels(image_files)  # Generate labels if missing

    processed_images = []

    for img_name in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(IMAGE_PATH, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is None:
            print(f"⚠ Warning: Could not read {img_path}. Skipping...")
            continue

        img = cv2.resize(img, IMAGE_SIZE)  # Resize
        img = img / 255.0  # Normalize to [0,1]
        processed_images.append(img)

    # Convert to NumPy array
    processed_images = np.array(processed_images, dtype=np.float32)

    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save processed data
    np.save(os.path.join(OUTPUT_PATH, "data.npy"), processed_images)
    np.save(os.path.join(OUTPUT_PATH, "labels.npy"), labels)

    print(f"✅ Processed {len(processed_images)} images!")
    print("✅ Labels saved!")

if __name__ == "__main__":
    preprocess_data()
