import torch
import torch.nn as nn
import cv2
class VisionCNN(nn.Module):
    """
    Entrada: (B,3,H,W) float o uint8; también acepta (B,H,W,3).
    Salida:  (B, feat_dim)
    Independiente de H y W gracias a GlobalAvgPool.
    """
    def __init__(self, in_channels=12, feat_dim=256, **kwargs):
        # compat: si en algún lugar quedó out_dim, lo aceptamos como alias
        if "out_dim" in kwargs:
            feat_dim = kwargs.pop("out_dim")
        assert not kwargs, f"Parámetros no esperados: {list(kwargs.keys())}"
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=2),  # downsample fuerte
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # Global Average Pooling -> (B,64,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),              # (B,64)
            nn.Linear(64, feat_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x: torch.Tensor):
        # normaliza si viene uint8
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        # NHWC -> NCHW si hace falta
        if x.dim() == 4 and x.size(1) not in (1, 3) and x.size(-1) in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = self.gap(x)
        x = self.head(x)
        return x