import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import streamlit as st
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
#from deepskin.model import build_efficientnetb3_unet  # adjust name based on repo
import shutil
from pathlib import Path

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tensorflow as tf
from deepskin.model import deepskin_model

sys.path.append(os.path.abspath("Deepskin"))

# Import your utilities
from utils import (
    segment_wound_with_deepskin,
    calculate_healing,
    overlay_masks,
    save_wound_data,
    load_wound_data
)

# ===========================
# Page Setup + Custom Styles
# ===========================
st.set_page_config(page_title="Wound Healing Tracker", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(120deg, #e0f7fa, #e8f5e9);
        color: #004d40;
    }
    .title {
        font-size: 40px;
        text-align: center;
        font-weight: bold;
        color: #00695c;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 40px;
    }
    .card {
        background-color: white;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Wound Healing Tracker</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Monitor wound recovery progress with AI assistance</div>", unsafe_allow_html=True)

# ===========================
# Load Models
# ===========================
#Dummy segmentation model (replace with your DeepSkin)
# class UNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # TODO: replace with actual DeepSkin U-Net layers
#     def forward(self, x):
#         return x  # placeholder

# segment_model = UNet()
sys.path.append(os.path.abspath("Deepskin"))
import tensorflow as tf

#segment_model = tf.keras.models.load_model("Deepskin/checkpoints/efficientnetb3_deepskin_semantic.h5",compile=False)
# segment_model.load_state_dict(torch.load("models/unet_model.pth", map_location='cpu'))

#segment_model = build_efficientnetb3_unet()  # instantiate architecture
#segment_model.load_weights("Deepskin/checkpoints/efficientnetb3_deepskin_semantic.h5")
#segment_model.trainable = False
#segment_model.eval()

# instantiate architecture exactly as the repo defines it
segment_model = deepskin_model(verbose=False)

# load the weights-only HDF5 checkpoint (weights map to layer names found inside the file)
weights_path = "Deepskin/checkpoints/efficientnetb3_deepskin_semantic.h5"
segment_model.load_weights(weights_path)

# disable training mode (TF way)
segment_model.trainable = False

# Healing prediction model
from models.healing.predictor import HealingPredictor
healing_model = HealingPredictor()
healing_model.load_state_dict(torch.load("models/healing_predictor.pth", map_location="cpu"))
healing_model.eval()

# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966489.png", width=120)
    st.markdown("### Settings")
    patient_id = st.text_input("Patient ID", value="wound_001")
    day = st.number_input("Day of Healing", min_value=0, value=0, step=1)
    st.markdown("---")
    st.markdown("Upload your wound images sequentially to track healing progress over time.")

# ===========================
# Upload & Process Section
# ===========================
st.header("Upload Wound Image")

st.markdown('<p style="color: black; font-weight: bold;">Choose an image</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

images, masks, days = load_wound_data(patient_id)

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Uploaded Image - Day {day}", width=200)
    
    with st.spinner("Segmenting wound area using DeepSkin model..."):
        mask = segment_wound_with_deepskin(img)
    
    save_wound_data(patient_id, img, mask, day)
    st.success(f"Image for day {day} saved successfully!")

# Reload updated data
images, masks, days = load_wound_data(patient_id)
st.markdown("</div>", unsafe_allow_html=True)

if st.sidebar.button("Reset app (clear cache & data)"):
    # Clear Streamlit caches
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    # Clear session state
    try:
        st.session_state.clear()
    except Exception:
        pass

    # Delete saved wound data folder (full reset)
    base = Path("wound_data")
    if base.exists():
        shutil.rmtree(base)

    # Reload the app
    st.rerun()

# ===========================
# Progress Section
# ===========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("Healing Analysis")

if len(images) > 1:
    first_mask = masks[0]
    healed_percents, overlay_imgs = [], []

    for i in range(1, len(images)):
        prev_mask, curr_mask = masks[i-1], masks[i]
        healed_percent = calculate_healing(first_mask, curr_mask)
        healed_percents.append(healed_percent)
        overlay = overlay_masks(images[i], prev_mask, curr_mask)
        overlay_imgs.append(overlay)

    # Show Overlays
    for i, overlay in enumerate(overlay_imgs):
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"Overlay: Day {days[i+1]} vs Day {days[i]}", width=200)
        st.markdown(f"**Healing since start:** {healed_percents[i]:.2f}%")

    # Healing Trend Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, [0] + healed_percents, marker="o", color="#00796b", label="Observed Healing")

    # === Replace this block ===
    # last_mask = masks[-1]
    # input_tensor = torch.tensor(last_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # # with torch.no_grad():
    # predicted_healing = healing_model(input_tensor).item()
    # === End replacement ===

    # take last mask (H,W) uint8 (0/255)
    last_mask = masks[-1]
    if last_mask is None:
        raise RuntimeError("No mask found for prediction")

    # normalize to [0,1] float
    mask_f = (last_mask > 0).astype(np.float32)  # values 0.0 or 1.0

    # determine expected input channels if possible
    expected_in_ch = None
    try:
        # common pattern: backbone conv1 exists (ResNet)
        expected_in_ch = healing_model.backbone.conv1.weight.shape[1]
    except Exception:
        # fallback to 3 (safe default for ResNet)
        expected_in_ch = 3

    # create multi-channel input to match expected_in_ch
    if expected_in_ch == 3:
        img = np.stack([mask_f, mask_f, mask_f], axis=2)  # H,W,3
    elif expected_in_ch == 1:
        img = mask_f[..., None]  # H,W,1
    else:
        # unusual: replicate channels to match expected channels
        img = np.repeat(mask_f[..., None], expected_in_ch, axis=2)

    # resize to model's spatial input size
    # common for torchvision ResNet is 224x224 — change if your predictor uses a different size
    MODEL_SPATIAL = 224
    img_resized = cv2.resize(img, (MODEL_SPATIAL, MODEL_SPATIAL), interpolation=cv2.INTER_LINEAR)

    # convert to tensor: shape (1, C, H, W)
    tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).unsqueeze(0).float()

    # If backbone is ImageNet pretrained, apply ImageNet normalization (only for 3-channel case)
    device = next(healing_model.parameters()).device
    tensor = tensor.to(device)
    if expected_in_ch >= 3:
        # use first 3 channels if there are more
        if tensor.shape[1] > 3:
            tensor = tensor[:, :3, :, :]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    else:
        # for single-channel models you can optionally normalize by mean/std = (0.5,0.5)
        tensor = (tensor - 0.5) / 0.5

    # run prediction
    healing_model.eval()
    with torch.no_grad():
        out = healing_model(tensor)
        # if the model returns a tensor scalar or size (1,) etc.
        try:
            predicted_healing = float(out.item())
        except Exception:
            # if model returns a tensor, try reducing
            predicted_healing = float(out.squeeze().cpu().numpy())

    # now predicted_healing is your scalar prediction


    ax.axhline(predicted_healing, color="orange", linestyle="--", label=f"Predicted Healing ({predicted_healing:.2f}%)")

    ax.set_xlabel("Day")
    ax.set_ylabel("% Healing")
    ax.set_title("Wound Healing Progress")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Healing Summary
    st.subheader("Healing Summary")
    df = pd.DataFrame({"Day": days[1:], "% Healed": healed_percents})
    st.dataframe(df.style.highlight_max(axis=0, color="#c8e6c9"))

    st.success(f"Predicted Healing Progress (AI Model): {predicted_healing:.2f}%")

else:
    st.info("Upload at least two images (different days) to visualize healing progress.")
st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# Footer
# ===========================
st.markdown("""
---
*Tip:* Upload wound images periodically (e.g., every few days) to track healing trends.  
Built with using DeepSkin + Streamlit.
""")