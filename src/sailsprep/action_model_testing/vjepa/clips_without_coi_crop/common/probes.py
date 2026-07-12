import torch
import torch.nn as nn


class AttentiveProbe(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, num_queries=1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn   = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm         = nn.LayerNorm(embed_dim)
        self.classifier   = nn.Linear(embed_dim * num_queries, num_classes)

    def forward(self, x):
        B       = x.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        out, _  = self.cross_attn(queries, x, x)
        out     = self.norm(out).reshape(B, -1)
        return self.classifier(out)
