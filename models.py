# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) CLASSIFIER (matches checkpoint EXACTLY)
# ============================================================
class Classifier(nn.Module):
    def __init__(self, input_dim=2000, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),       # net.0
            nn.Linear(256, 128), nn.ReLU(),            # net.3
            nn.Linear(128, 64), nn.ReLU()              # net.6
        )
        self.fc = nn.Linear(64, n_classes)             # fc

    def forward(self, x, return_embed=False):
        emb = self.net(x)                             # 64-dim embedding
        logits = self.fc(emb)
        if return_embed:
            return logits, emb
        return logits


# ============================================================
# 2) TRIPLET MODEL (MUST ACCEPT 64-dim INPUT)
# ============================================================
class TripletNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.fc(x)       # Output: 32-dim meta features


# ============================================================
# 3) META DETECTOR (uses 32-dim features)
# ============================================================
class MetaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)       # Output: single detection score
