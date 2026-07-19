import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np

CLASSES = [
    "Bud Root Dropping",
    "Bud Rot",
    "Gray Leaf Spot",
    "Leaf Rot",
    "Stem Bleeding",
]
PROCESSED_DIR = "data/processed"
FEATURES_DIR  = "features"
BATCH_SIZE    = 32

TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


class FolderDataset(Dataset):
    """Load ảnh từ data/processed/{split}/{class}/*.jpg"""
    def __init__(self, split: str):
        self.items = []
        for i, cls in enumerate(CLASSES):
            folder = Path(PROCESSED_DIR) / split / cls
            if not folder.exists():
                print(f"  [WARN] Không thấy: {folder}")
                continue
            imgs = (list(folder.glob("*.jpg"))
                  + list(folder.glob("*.jpeg"))
                  + list(folder.glob("*.png")))
            if len(imgs) == 0:
                print(f"  [WARN] Không có ảnh trong: {folder}")
            for p in imgs:
                self.items.append((str(p), i))
        print(f"  [{split}] tổng {len(self.items)} ảnh")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return TRANSFORM(img), label


def build_backbone(name: str):
    if name == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Identity()          
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Identity()          
    else:
        import timm
        m = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0
        )                             
    return m.eval()


def extract_split(backbone, split: str, device) -> tuple:
    ds = FolderDataset(split)
    if len(ds) == 0:
        raise RuntimeError(
            f"Split '{split}' không có ảnh nào.\n"
            f"Kiểm tra thư mục: {PROCESSED_DIR}/{split}/"
        )
    dl = DataLoader(ds, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for imgs, ys in dl:
            f = backbone(imgs.to(device)).cpu().numpy()
            feats.append(f)
            labels.append(ys.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main(backbone_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Backbone : {backbone_name}")
    print(f"Device   : {device}")
    print(f"Data dir : {PROCESSED_DIR}")

    for split in ["train", "val", "test"]:
        p = Path(PROCESSED_DIR) / split
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy: {p}\n"
                f"Hãy chạy prepare_data.py trước."
            )

    backbone = build_backbone(backbone_name).to(device)
    Path(FEATURES_DIR).mkdir(exist_ok=True)

    out = {}
    for split in ["train", "val", "test"]:
        print(f"\nExtracting [{split}]...")
        X, y = extract_split(backbone, split, device)
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        print(f"  features shape: {X.shape}")

    save_path = f"{FEATURES_DIR}/features_{backbone_name}.npz"
    np.savez(save_path, **out)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "vit_small"],
    )
    args = ap.parse_args()
    main(args.backbone)