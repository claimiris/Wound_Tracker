import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from models.healing.predictor import HealingPredictor

# ----- Config -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "healing_data.csv"
SAVE_PATH = "models/healing_predictor.pth"
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-4

# Set NUM_WORKERS to 0 while debugging on Windows. Increase later (1,2,4) if stable.
NUM_WORKERS = 0
PIN_MEMORY = False  # set True only when using GPU

# ----- Dataset -----
class WoundDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.get('img_path') or row.get('image_path')  # tolerate either name
        if not isinstance(img_path, str) or not img_path:
            raise RuntimeError(f"Bad image path at row {idx}: {img_path}")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Raise a descriptive error so the DataLoader worker (or main) shows which file failed
            raise RuntimeError(f"Error opening image '{img_path}' (row {idx}): {e}")

        try:
            image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error transforming image '{img_path}' (row {idx}): {e}")

        # target column tolerant to 'healing_pct' or 'target'
        if 'healing_pct' in row:
            target_val = row['healing_pct']
        elif 'target' in row:
            target_val = row['target']
        else:
            raise RuntimeError("CSV must contain 'healing_pct' or 'target' column")

        try:
            target = float(target_val) / 100.0
        except Exception as e:
            raise RuntimeError(f"Bad target value at row {idx}: {target_val} ({e})")

        return image, torch.tensor(target, dtype=torch.float32)

# ----- Training function -----
def run_training():
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    # dataset + loader
    dataset = WoundDataset(CSV_PATH)

    # quick sanity check (first few samples) to surface dataset errors before launching full training
    print("Running a quick dataset sanity check (first 4 samples)...")
    try:
        for i in range(min(4, len(dataset))):
            img, tgt = dataset[i]
            print(f"  sample {i}: img shape {tuple(img.shape)}, target {float(tgt):.4f}")
    except Exception as e:
        print("Dataset sanity check failed:", e)
        raise

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # model / optimizer
    model = HealingPredictor(pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        seen = 0
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(imgs).squeeze(1)  # shape (B,)
            loss = F.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            seen += bs

        avg_loss = total_loss / seen if seen > 0 else float("nan")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train MSE: {avg_loss:.6f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

# ----- Windows multiprocessing protection -----
if __name__ == "__main__":
    # Windows safety for multiprocessing DataLoader
    from multiprocessing import freeze_support
    freeze_support()

    # If you want to explicitly set start method (optional on Windows)
    try:
        import multiprocessing as mp
        # mp.set_start_method('spawn', force=True)  # default on Windows is 'spawn'
    except Exception:
        pass

    run_training()

