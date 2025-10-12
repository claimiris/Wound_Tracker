import cv2
import numpy as np
import torch
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
from deepskin.model import deepskin_model as DeepskinUNet

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

try:
    # deepskin exposes a simple function (README example)
    # after pip install -e Deepskin or sys.path append
    from deepskin import wound_segmentation
except Exception as e:
    # gracefully fallback or raise a helpful error
    raise ImportError("Could not import deepskin. Did you install the Deepskin repo or append it to PYTHONPATH? "
                      "Run `pip install -e Deepskin` or `sys.path.append('Deepskin')`") from e

_MODEL = None

def get_deepskin_model(checkpoint_path="models/deepskin _checkpoint.pth", device='cpu'):
    global _MODEL
    if _MODEL is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepskinUNet()   # if constructor needs args, pass them (check model file)
        ckpt = torch.load(checkpoint_path, map_location=device)
        # common patterns for checkpoint dicts:
        if isinstance(ckpt, dict):
            sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        else:
            sd = ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        _MODEL = model
    return _MODEL

# ------------------------
# Segmentation
# ------------------------
def preprocess(image, size=(256,256), device='cpu'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2,0,1))
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    return tensor

#def segment_wound(model, image):
    input_tensor = preprocess(image)
    with torch.no_grad():
        mask = model(input_tensor)
    mask = mask.squeeze().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    return mask

#def segment_wound_with_deepskin(image_bgr):
    """
    image_bgr: OpenCV BGR image (H,W,3)
    returns: binary_mask (H,W) uint8 0/255
    """
    model = get_deepskin_model()
    # --- use the repo's preprocessing exactly if available (look in deepskin/imgproc.py) ---
    # quick generic preprocess (may differ from repo's)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256,256))  # use repo size
    inp = img_resized.astype('float32') / 255.0
    inp = np.transpose(inp, (2,0,1))[None]  # shape (1,3,H,W)
    inp_tensor = torch.from_numpy(inp)
    with torch.no_grad():
        out = model(inp_tensor)   # repo may return logits or mask; inspect model forward
        # assume it returns logits (1,H,W) or (1,1,H,W)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out_np = out.squeeze().cpu().numpy()
        # if logits, apply sigmoid
        if out_np.max() > 1.0 or out_np.min() < 0.0:
            out_np = 1.0 / (1.0 + np.exp(-out_np))  # sigmoid
        mask_small = (out_np > 0.5).astype('uint8') * 255
    mask = cv2.resize(mask_small, (image_bgr.shape[1], image_bgr.shape[0]))
    return mask

#def segment_wound(image_bgr, device=None):
    model = get_deepskin_model(device=device)
    device = device or next(model.parameters()).device
    inp_tensor = preprocess(image_bgr, device=device)
    
    with torch.no_grad():
        out = model(inp_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        mask_np = out.squeeze().cpu().numpy()
        # apply sigmoid if output is not in [0,1]
        if mask_np.max() > 1.0 or mask_np.min() < 0.0:
            mask_np = 1.0 / (1.0 + np.exp(-mask_np))
        mask_small = (mask_np > 0.5).astype(np.uint8) * 255
    
    # resize to original image size
    mask = cv2.resize(mask_small, (image_bgr.shape[1], image_bgr.shape[0]))
    return mask

def segment_wound_with_deepskin(image_bgr):
    """
    Minimal, robust segmentation using Deepskin's TF helper:
      - Accepts OpenCV BGR image (H,W,3)
      - Calls wound_segmentation(img=rgb) from Deepskin if available
      - Converts various possible outputs into a binary mask (H,W) dtype uint8 with 0/255
    """
    # sanity checks
    if 'wound_segmentation' not in globals() or not callable(globals()['wound_segmentation']):
        raise ImportError(
            "Deepskin TF helper `wound_segmentation` not available. "
            "Install the Deepskin repo (e.g. `pip install -e Deepskin`) or ensure `sys.path` includes the repo root."
        )

    # convert to RGB because Deepskin expects RGB images
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # call the repo helper
    raw_mask = wound_segmentation(img=rgb)
    mask_arr = np.array(raw_mask)

    # handle common return shapes
    if mask_arr.ndim == 3 and mask_arr.shape[2] in (2, 3):
        # could be probabilities (H,W,C) or one-hot channels -> argmax to label map
        label_map = np.argmax(mask_arr, axis=2).astype(np.uint8)

        # heuristic to choose wound label:
        # many semantic setups use label 1 for wound; prefer 1 if present, otherwise pick 0
        unique_labels = np.unique(label_map)
        wound_label = 1 if 1 in unique_labels else 0
        wound_mask = (label_map == wound_label).astype(np.uint8) * 255

    elif mask_arr.ndim == 2:
        # Could be a 2D label map or a binary mask
        if mask_arr.dtype == np.uint8 and mask_arr.max() > 1:
            # probably 0/255 mask
            wound_mask = (mask_arr > 127).astype(np.uint8) * 255
        else:
            # assume label map with wound label == 1
            wound_mask = (mask_arr == 1).astype(np.uint8) * 255

    else:
        # Unexpected shape => fallback: grayscale threshold of the RGB input
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, wound_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # ensure mask has the original image size
    h, w = image_bgr.shape[:2]
    if wound_mask.shape != (h, w):
        wound_mask = cv2.resize(wound_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return wound_mask.astype(np.uint8)
# ------------------------
# Healing Calculation
# ------------------------
def calculate_healing(mask_initial, mask_current):
    area_initial = np.sum(mask_initial > 0)
    area_current = np.sum(mask_current > 0)
    if area_initial == 0:
        return 0
    healed_percent = ((area_initial - area_current) / area_initial) * 100
    healed_percent = np.clip(healed_percent, 0, 100)
    return healed_percent

# ------------------------
# Overlay Masks
# ------------------------
#def overlay_masks(image, mask_prev, mask_current):
    overlay = image.copy()
    overlay[mask_prev>0] = [255,0,0]   # Previous = red
    overlay[mask_current>0] = [0,255,0] # Current = green
    return overlay
def overlay_masks(image, mask_prev, mask_current, alpha=0.5):
    """
    Create an overlay showing previous mask in red and current mask in green.
    Returns a BGR image (same dtype as input image).

    This function is defensive:
      - ensures masks are 2D uint8 (0/255),
      - resizes masks to image size if needed,
      - uses alpha blending to combine colors.
    """
    # copy input to avoid mutating caller data
    overlay = image.copy()
    h, w = image.shape[:2]

    # ensure masks are numpy arrays
    mask_prev = np.asarray(mask_prev) if mask_prev is not None else np.zeros((h, w), dtype=np.uint8)
    mask_current = np.asarray(mask_current) if mask_current is not None else np.zeros((h, w), dtype=np.uint8)

    # if masks are single-channel images with 3 dims (H,W,1), squeeze them
    if mask_prev.ndim == 3 and mask_prev.shape[2] == 1:
        mask_prev = mask_prev[:, :, 0]
    if mask_current.ndim == 3 and mask_current.shape[2] == 1:
        mask_current = mask_current[:, :, 0]

    # resize masks to match image if necessary
    if mask_prev.shape[:2] != (h, w):
        mask_prev = cv2.resize(mask_prev, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask_current.shape[:2] != (h, w):
        mask_current = cv2.resize(mask_current, (w, h), interpolation=cv2.INTER_NEAREST)

    # ensure dtype and boolean
    mask_prev_bool = (mask_prev > 0)
    mask_curr_bool = (mask_current > 0)

    # create a colored overlay copy
    colored = image.copy()

    # paint previous mask red (BGR)
    colored[mask_prev_bool] = [0, 0, 255]   # red
    # paint current mask green
    colored[mask_curr_bool] = [0, 255, 0]   # green

    # alpha blend original image and colored overlays
    blended = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)

    return blended
#using alpha blending
#def overlay_masks(image, mask_prev, mask_current, alpha=0.5):
    overlay = image.copy()
    overlay[mask_prev>0] = [255,0,0]
    overlay[mask_current>0] = [0,255,0]
    return cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

# ------------------------
# Save Data
# ------------------------
def save_wound_data(patient_id, image, mask, day):
    folder = f'wound_data/{patient_id}/images'
    mask_folder = f'wound_data/{patient_id}/masks'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    
    # Save image
    image_path = f'{folder}/day_{day}.png'
    cv2.imwrite(image_path, image)
    
    # Save mask
    mask_path = f'{mask_folder}/day_{day}_mask.png'
    cv2.imwrite(mask_path, mask)
    
    # Update metadata
    metadata_file = f'wound_data/{patient_id}/metadata.csv'
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
    else:
        df = pd.DataFrame(columns=['day','image','mask','timestamp'])
    df = pd.concat([df, pd.DataFrame([{
    'day': day,
    'image': image_path,
    'mask': mask_path,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}])], ignore_index=True)
    df.to_csv(metadata_file, index=False)

# ------------------------
# Load Data
# ------------------------
def load_wound_data(patient_id):
    metadata_file = f'wound_data/{patient_id}/metadata.csv'
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        images = [cv2.imread(p) for p in df['image']]
        masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in df['mask']]
        days = list(df['day'])
        return images, masks, days
    return [], [], []