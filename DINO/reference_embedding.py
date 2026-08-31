import torch
import torch.nn as nn
import math

class ReferencePointEmbedding(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 temperature=10000):
        super().__init__()

        assert hidden_dim % 4 == 0

        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.coord_dim = hidden_dim // 4

    def forward(self, reference_boxes):
        # [B, Q, 4]
        # x,y,w,h

        dim_t = torch.arange(
            self.coord_dim,
            dtype=torch.float32,
            device=reference_boxes.device
        )

        dim_t = self.temperature ** (
            2 * torch.div(
                dim_t,
                2,
                rounding_mode="floor"
            ) / self.hidden_dim
        )

        reference_boxes = (
            reference_boxes * 2 * math.pi
        )

        # [B,Q, 4,1]
        pos = reference_boxes.unsqueeze(-1) / dim_t

        # [B,Q,4,256]
        pos = torch.cat(
            [
                pos.sin(),
                pos.cos()
            ],
            dim=-1
        )

        # [B, Q, 4*256]
        pos = pos.flatten(
            start_dim=2
        )

        return pos