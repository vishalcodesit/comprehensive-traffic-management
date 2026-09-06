"""
Optional: fine-tunes a MobileNetV3-Small binary classifier (emergency vs.
normal vehicle) to ensemble with the heuristics in emergency_classifier.py.

Not required to run the project -- emergency_classifier.py works without a
trained checkpoint (heuristics only). Use this once you have a labeled
image-crop dataset, e.g. a Kaggle "emergency vehicle identification" set,
laid out as:

    data/emergency_vehicles/train/emergency/*.jpg
    data/emergency_vehicles/train/normal/*.jpg
    data/emergency_vehicles/val/emergency/*.jpg
    data/emergency_vehicles/val/normal/*.jpg

Run:  python -m src.detection.train_emergency_cnn
"""
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DATASET_DIR = os.path.join(config.DATA_DIR, "emergency_vehicles")
OUT_PATH = os.path.join(config.MODELS_DIR, "emergency_cnn.pt")


def run(epochs=10, batch_size=32, lr=1e-4):
    train_dir, val_dir = os.path.join(DATASET_DIR, "train"), os.path.join(DATASET_DIR, "val")
    if not os.path.isdir(train_dir):
        print(f"No dataset found at {DATASET_DIR}. See this file's docstring for the "
              f"expected layout. Skipping -- the project runs fine on heuristics alone.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = T.Compose([
        T.Resize((160, 160)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((160, 160)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Classes: {train_ds.classes} (index 1 must be 'normal' > "
          f"'emergency' alphabetically is index 0 -- emergency_classifier.py assumes index 1 = emergency)")

    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for p in net.features.parameters():
        p.requires_grad = False
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    net.to(device)

    optimizer = torch.optim.Adam(net.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()

        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = net(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), OUT_PATH)

    print(f"Best val_acc={best_acc:.3f}. Saved to {OUT_PATH}")


if __name__ == "__main__":
    run()
