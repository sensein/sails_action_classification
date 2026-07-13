import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Simplest baseline: mean-pool all patch tokens -> single Linear layer.
    No nonlinearity. Purely tests linear separability of VJEPA features.
    """
    def __init__(self, embed_dim, num_classes, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):           # x: [B, N, D]
        return self.head(self.norm(x.mean(dim=1)))


class MLPSmallProbe(nn.Module):
    """
    Mean-pool -> LayerNorm -> one hidden layer (512) -> GELU -> Dropout -> Linear.
    Adds nonlinearity over LinearProbe with minimal parameters.
    """
    def __init__(self, embed_dim, num_classes, hidden=512, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):           # x: [B, N, D]
        return self.net(x.mean(dim=1))


class MLPLargeProbe(nn.Module):
    """
    Mean-pool -> LayerNorm -> 1024 -> GELU -> 512 -> GELU -> Dropout -> Linear.
    Deeper MLP; more capacity to learn non-linear feature combinations.
    """
    def __init__(self, embed_dim, num_classes, dropout=0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):           # x: [B, N, D]
        return self.net(x.mean(dim=1))
