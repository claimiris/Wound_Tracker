# prepare_dataset.py
"""
Prepare healing_data.csv using the deepskin tensorflow segmentation model.

Assumptions / notes:
- deepskin.deepskin_model() returns a tf.keras.Model that outputs softmax with 3 channels.
- We try to load segmentation weights from a few standard locations (SavedModel dir or .h5).
- The script computes wound area per image (in pixels at model input resolution) and
  healing_pct = 100 * (1 - area / base_area) where base_area is the first image for each wound.
- By default we assume wound class is channel index 1 (middle channel). Change WOUND_CLASS_INDEX if needed.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# import model and constants from deepskin package
from deepskin.model import deepskin_model
from deepskin.constants import IMG_SIZE

# --- USER-CONFIGURABLE --- #
DATA_DIR = "woundsdata"                   # folder containing wound subfolders
OUTPUT_CSV = "healing_data.csv"
SEG_WEIGHTS_DIR = "models/unet_model"     # try SavedModel dir first
SEG_WEIGHTS_H5 = "models/unet_model.h5"
SEG_WEIGHTS_H5_2 = "Deepskin/checkpoints/efficientnetb3_deepskin_semantic.h5"
WOUND_CLASS_INDEX = 1                     # which softmax channel corresponds to wound (0/1/2)
USE_PROB_THRESHOLD = False                # if True, use probability thresholding instead of argmax
PROB_THRESHOLD = 0.5
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BATCH_PREDICT = False                     # if True, will predict images in batches (saves overhead). For small dataset False is fine.
BATCH_SIZE = 8
# ------------------------- #

# build model
print("Building deepskin segmentation model...")
tf.keras.backend.clear_session()
seg_model = deepskin_model(verbose=False)

# try to load weights from common places
loaded_weights = False
if os.path.isdir(SEG_WEIGHTS_DIR):
    try:
        # attempt to load a SavedModel directory
        print(f"Attempting to load SavedModel from '{SEG_WEIGHTS_DIR}' ...")
        seg_model = tf.keras.models.load_model(SEG_WEIGHTS_DIR, compile=False)
        loaded_weights = True
        print("Loaded SavedModel directory.")
    except Exception as e:
        print(f"Could not load SavedModel dir: {e}. Will try other formats.")
if (not loaded_weights) and os.path.isfile(SEG_WEIGHTS_H5):
    try:
        print(f"Attempting to load weights from '{SEG_WEIGHTS_H5}' ...")
        seg_model.load_weights(SEG_WEIGHTS_H5)
        loaded_weights = True
        print("Loaded weights from .h5 file.")
    except Exception as e:
        print(f"Could not load .h5 weights: {e}.")
if (not loaded_weights) and os.path.isfile(SEG_WEIGHTS_H5_2):
    try:
        print(f"Attempting to load weights from '{SEG_WEIGHTS_H5_2}' ...")
        seg_model.load_weights(SEG_WEIGHTS_H5_2)
        loaded_weights = True
        print("Loaded weights from alternate .h5 file.")
    except Exception as e:
        print(f"Could not load alternate .h5 weights: {e}.")

if not loaded_weights:
    warnings.warn(
        "No segmentation weights loaded. The model will run with random weights and predictions will be meaningless. "
        "Place your SavedModel folder at 'models/unet_model' or a weights file at 'models/unet_model.h5'."
    )

seg_model.trainable = False
print("Segmentation model ready. IMG_SIZE =", IMG_SIZE)

def preprocess_for_segmentation(pil_image, target_size=IMG_SIZE):
    """Resize PIL Image to (target_size,target_size) and return preprocessed numpy array."""
    img = pil_image.resize((target_size, target_size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # efficientnet preprocess_input expects float array and will scale appropriately
    arr = eff_preprocess(arr)
    return arr

def predict_mask_for_image(pil_image):
    """
    Returns:
      - mask_probs: np.array shape (H,W,3) of softmax probabilities (sum to 1 across channels)
      - mask_labels: np.array shape (H,W) of integer class labels (argmax)
    """
    arr = preprocess_for_segmentation(pil_image, target_size=IMG_SIZE)
    inp = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    preds = seg_model.predict(inp, verbose=0)  # (1,H,W,3)
    preds = np.asarray(preds[0])  # (H,W,3)
    # sanity check: if preds are logits not probabilities, softmax may already have been used in model (it is in code).
    # We'll assume preds are probabilities (softmax output). If they are not, you could apply softmax here.
    labels = np.argmax(preds, axis=-1).astype(np.int32)  # (H,W)
    return preds, labels

records = []
wound_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

if len(wound_folders) == 0:
    print("No wound subfolders found in", DATA_DIR)
    sys.exit(1)

for wound_id in wound_folders:
    folder = os.path.join(DATA_DIR, wound_id)
    image_files = sorted([f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in ALLOWED_EXTS])
    if len(image_files) == 0:
        warnings.warn(f"No images in {folder}, skipping.")
        continue

    base_area = None
    # process each image in the wound folder with per-wound day index
    for day_idx, img_file in enumerate(image_files):
        img_path = os.path.join(folder, img_file)
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Could not open {img_path}: {e}. Skipping.")
            continue

        # predict mask
        probs, labels = predict_mask_for_image(pil)

        if USE_PROB_THRESHOLD:
            # use probability for wound class
            wound_prob_map = probs[..., WOUND_CLASS_INDEX]
            mask_binary = (wound_prob_map >= PROB_THRESHOLD).astype(np.uint8)
            area = int(mask_binary.sum())
        else:
            # use argmax labels
            mask_binary = (labels == WOUND_CLASS_INDEX).astype(np.uint8)
            area = int(mask_binary.sum())

        if base_area is None:
            # first image per wound -> baseline
            base_area = area if area > 0 else 1
            if area == 0:
                warnings.warn(f"Base area for wound {wound_id} image {img_file} is zero. Using base_area=1 to avoid divide-by-zero.")

        healing_pct = 100.0 * (1.0 - (area / float(base_area)))
        records.append({
            "wound_id": wound_id,
            "img_path": img_path,
            "day": int(day_idx),
            "wound_area": int(area),
            "healing_pct": float(healing_pct)
        })
        print(f"{wound_id} | day {day_idx} | file {img_file} | area {area} | heal {healing_pct:.2f}%")

# write CSV
df = pd.DataFrame(records, columns=["wound_id", "img_path", "day", "wound_area", "healing_pct"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
