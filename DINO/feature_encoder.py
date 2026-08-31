import torch
import torch.nn as nn
from positional_encoding import PositionEmbeddingSine

class MultiScaleFeatureFlatten(nn.Module):
    def __init__(self,
                 hidden_dim=256):
        super().__init__()

        self.position_embedding = PositionEmbeddingSine(
            hidden_dim=hidden_dim
        )

    def forward(self,features):

        src_flatten = []
        pos_flatten = []
        spatial_shapes = []

        for feature in features:

            batch_size,channels, height,width = feature.shape

            #Store spatial dimensions
            spatial_shapes.append(
                (height,width)
            )

            #Positional encoding
            pos = self.position_embedding(feature)

            #Flatten spatial dimensions
            #[B,C,H,W] to [B,H*W,C]

            feature = feature.flatten(
                2
            ).transpose(1,2)

            pos = pos.flatten(
                2
            ).transpose(1,2)

            src_flatten.append(feature)
            pos_flatten.append(pos)

        #Concatenate all feature levels
        src_flatten = torch.cat(
            src_flatten,
            dim=1
        )

        pos_flatten = torch.cat(
            pos_flatten,
            dim=1
        )

        #[num_levels, 2]
        spatial_shapes = torch.tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device
        )

        level_start_index = torch.cat([
            torch.tensor([0],
                         dtype=torch.long,
                         device=spatial_shapes.device),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])

        return (
            src_flatten,
            pos_flatten,
            spatial_shapes,
            level_start_index
        )

        

