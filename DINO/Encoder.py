import torch
import torch.nn as nn
from Deformable_Attention import MultiScaleDeformableAttention,get_reference_points

class DeformableEncoderLayer(nn.Module):
    def __init__(self,
                 hidden_dims=256,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()

        self.self_attn = MultiScaleDeformableAttention(
            hidden_dims=hidden_dims,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points
        )

        #First Normalization
        self.norm1 = nn.LayerNorm(hidden_dims)

        #Feed Forward Network
        self.feed_forward1 = nn.Linear(
            hidden_dims,
            dim_feedforward
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.feed_forward2 = nn.Linear(
            dim_feedforward,
            hidden_dims
        )

        #Second Normalization
        self.norm2 = nn.LayerNorm(hidden_dims)

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index):

        #Add Positional information to query
        query = src + pos

        #Deformable Self Attention
        src2 = self.self_attn(
            query=query,
            reference_points= reference_points,
            value = src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        #Residual connection + normalization
        src = src + self.dropout(src2)
        src = self.norm1(src)

        #Feed Forward Network
        src2 = self.feed_forward2(
            self.dropout(
                self.activation(
                    self.feed_forward1(src)
                )
            )
        )

        #Residual Connection + normalization
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

#Test
# B = 1
# HIDDEN_DIM = 64
# NUM_LEVELS = 3
# NUM_HEADS = 4
# NUM_POINTS = 2

# spatial_shapes = torch.tensor([
#     [8,8],
#     [4,4],
#     [2,2]
# ], dtype=torch.long)

# total_tokens = (
#     8 * 8 +
#     4 * 4 +
#     2 * 2
# )

# level_start_index = torch.tensor([
#     0,
#     64,
#     80
# ], dtype=torch.long)

# src = torch.randn(
#     B,
#     total_tokens,
#     HIDDEN_DIM
# )

# pos = torch.randn(
#     B,
#     total_tokens,
#     HIDDEN_DIM
# )

# reference_points = get_reference_points(
#     spatial_shapes=spatial_shapes,
#     batch_size=B,
#     device= src.device
# )



#Deformable Encoder as it stack multiple encoder layers
class DeformableEncoder(nn.Module):
    def __init__(self,
                 num_layers=6,
                 hidden_dim=256,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DeformableEncoderLayer(
                hidden_dims=hidden_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index):

        output = src

        for layer in self.layers:
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index
            )

        return output


# encoder_layer = DeformableEncoder(
#     hidden_dim=HIDDEN_DIM,
#     num_heads=NUM_HEADS,
#     num_levels=NUM_LEVELS,
#     num_points=NUM_POINTS
# )

# output = encoder_layer(
#     src,
#     pos,
#     reference_points,
#     spatial_shapes,
#     level_start_index
# )

# print("Output Shape:", output.shape)