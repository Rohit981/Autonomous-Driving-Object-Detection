import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 temperature=10000,
                 normalize=True,
                 scale=2*math.pi):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):

        batch_size,_, height,width = x.shape

        #Create coordinate grids
        y_embed = torch.arange(
            height,
            device=x.device
        ).view(1,height,1)

        x_embed = torch.arange(
            width,
            device=x.device
        ).view(1,1,width)

        y_embed = y_embed.expand(
            batch_size,
            height,
            width
        )

        x_embed = x_embed.expand(
            batch_size,
            height,
            width
        )

        if self.normalize:
            eps = 1e-6

            y_embed = (
                y_embed + 1
            ) / (
                height + eps
            ) * self.scale

            x_embed = (
                x_embed + 1
            ) / (
                width + eps
            ) * self.scale

        dim_t = torch.arange(
            self.hidden_dim // 2,
            dtype=torch.float32,
            device=x.device
        )

        dim_t = self.temperature ** (
            2 * torch.div(
                dim_t,
                2,
                rounding_mode="floor"
            ) / (self.hidden_dim // 2)
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:,:,:, None] / dim_t

        pos_x = torch.stack(
            (
                pos_x[:,:,:, 0:2].sin(),
                pos_x[:,:,:,1:2].cos()
            ),
            dim=4
        ).flatten(3)

        pos_y = torch.stack(
            (
                pos_y[:,:,:,0:2].sin(),
                pos_y[:,:,:,1:2].cos()
            ),
            dim=4
        ).flatten(3)

        #[B,H,W,256]
        pos = torch.cat(
            (pos_y, pos_x),
            dim=3
        )

        #[B,256,H,W]
        return pos.permute(
            0,
            3,
            1,
            2
        )