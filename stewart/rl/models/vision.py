import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """
    Entrada: (B,3,H,W) float [0,1]
    Salida:  feature vector (B, feat_dim)
    """
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=2),  # -> H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # -> H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64* (11*11), feat_dim),  # OJO: ajusta este 11*11 si tu img!=84x84
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: uint8 [0..255] o float; normalizamos a [0,1] y channel-first
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        # (B,H,W,3) -> (B,3,H,W) si llega NHWC
        if x.dim()==4 and x.shape[1] != 3:
            x = x.permute(0,3,1,2).contiguous()
        y = self.conv(x)
        return self.head(y)