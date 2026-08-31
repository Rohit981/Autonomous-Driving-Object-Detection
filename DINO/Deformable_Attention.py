import math

import torch
import torch.nn as nn
import torch.nn.functional as f

class MultiScaleDeformableAttention(nn.Module):
    def __init__(self,
                 hidden_dims=256,
                 num_levels=3,
                 num_heads=8,
                 num_points=4):
        super().__init__()

        assert hidden_dims % num_heads == 0
        self.hidden_dims = hidden_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.head_dim = hidden_dims // num_heads

        #Project input features to values
        self.value_proj = nn.Linear(
            hidden_dims,
            hidden_dims
        )

        #Predict Sampling offset
        #For Every Query:
        # num_heads
        # x num_levels
        # x num_points
        # x 2 coordinates
        self.sampling_offsets = nn.Linear(
            hidden_dims,
            num_heads*num_levels*num_points*2
        )

        #Predict attention weight for each sampling location
        self.attention_weights = nn.Linear(
            hidden_dims,
            num_heads*num_points*num_levels
        )

        #Final output projection
        self.output_proj = nn.Linear(
            hidden_dims,
            hidden_dims
        )

    def forward(
            self,
            query,
            reference_points,
            value,
            spatial_shapes,
            level_start_index):

        batch_size, num_queries, _ = query.shape

        #Projection Value
        value = self.value_proj(value)

        #Reshape value into attention head
        value = value.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        )

        #Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query)

        sampling_offsets = sampling_offsets.view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2
        )

        #Predict Attention weights
        attention_weights = self.attention_weights(query)

        attention_weights = attention_weights.view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points
        )

        attention_weights = f.softmax(
            attention_weights,
            dim=-1
        )

        reference_points = reference_points.unsqueeze(2).unsqueeze(4)
       

        #Calculate Sampling Locations
        normalizer = spatial_shapes[
            :, [1,0]
        ]

        sampling_locations = (
            reference_points
            +
            sampling_offsets / normalizer[
                None,None,None,:,None,:
            ]
        )

        #Sample Feature Maps
        output = self.sample_features(
            value,
            sampling_locations,
            attention_weights,
            spatial_shapes,
            level_start_index
        )

        #Merge Attention Heads
        output = output.reshape(
            batch_size,
            num_queries,
            self.hidden_dims
        )

        #Final Projection
        output = self.output_proj(output)

        return output

    #Bilinear Sampling
    def sample_features(
            self,
            value,
            sampling_locations,
            attention_weights,
            spatial_shapes,
            level_start_index):

        batch_size = value.shape[0]
        num_queries = sampling_locations.shape[1]

        output = torch.zeros(
            batch_size,
            num_queries,
            self.num_heads,
            self.head_dim,
            device=value.device,
            dtype=value.dtype
        )

        for level in range(self.num_levels):

            height = spatial_shapes[level,0]
            width = spatial_shapes[level,1]

            height = int(height.item())
            width = int(width.item())

            #Extract feature level
            start = level_start_index[level]

            if level + 1 < self.num_levels:
                end = level_start_index[level + 1]
            else:
                end = start + height*width

            value_level = value[
                :,
                start:end,
                :
            ]

            #Convert flattened feature map back to H X W
            value_level = value_level.view(
                batch_size,
                height,
                width,
                self.num_heads,
                self.head_dim
            )

            #Move heads before spatial dimensions
            value_level = value_level.permute(
                0,
                3,
                4,
                1,
                2
            )

            #Sampling locations for this level
            sampling_grid = sampling_locations[
                :,:,:,level
            ]

            #[B,Len_q, heads, points, 2]

            sampling_grid = sampling_grid.permute(
                0,
                2,
                1,
                3,
                4
            )

            #Convert [0,1] coordinated to [-1,1]
            sampling_grid = (
                sampling_grid * 2 - 1
            )

            #Sample each head
            for head in range(self.num_heads):

                feature = value_level[
                    :,head
                ]

                #[B,head_dim, H, W]
                grid = sampling_grid[
                    :,head
                ]

                #[B,len_q, points,2]
                sampled = f.grid_sample(
                    feature,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False
                )

                #Permute the array [B,head_dim, Len_q, points]
                sampled = sampled.permute(
                    0,
                    2,
                    3,
                    1
                )

                # [B,Len_q, points,head_dim]
                weights = attention_weights[
                    :,:,head,level
                ]

                #[B,len_q,points]
                weights = weights.unsqueeze(-1)

                output[:,:,head] += (
                    sampled*weights
                ).sum(dim=2)

        return output

def get_reference_points(
        spatial_shapes,
        batch_size,
        device):

    reference_points = []

    for height,width in spatial_shapes:

        height = int(height.item())
        width = int(width.item())

        y,x = torch.meshgrid(
            torch.arange(
                height,
                device=device
            ) + 0.5,

            torch.arange(
                width,
                device=device
            ) + 0.5,
            indexing="ij"
        )

        x = x.reshape(-1) / width
        y = y.reshape(-1) / height

        reference = torch.stack(
            (x,y),
            dim=-1
        )

        reference_points.append(reference)

    reference_points = torch.cat(
        reference_points,
        dim=0
    )

    reference_points = reference_points.unsqueeze(0)

    reference_points = reference_points.repeat(
        batch_size,
        1,
        1
    )

    reference_points = reference_points.unsqueeze(2)

    reference_points = reference_points.repeat(
        batch_size,
        1,
        spatial_shapes.shape[0],
        1
    )

    return reference_points


#Test
# B = 1
# NUM_QUERIES = 100
# HIDDEN_DIM = 64
# NUM_LEVELS = 3
# NUM_HEADS = 4
# NUM_POINTS = 2
# total_tokens = (
#     8 * 8 +
#     4 * 4 +
#     2 * 2
# )

# model = MultiScaleDeformableAttention(
#     hidden_dims=HIDDEN_DIM,
#     num_levels=NUM_LEVELS,
#     num_heads=NUM_HEADS,
#     num_points=NUM_POINTS
# )

# value = torch.randn(
#     B,
#     total_tokens,
#     HIDDEN_DIM
# )

# query = torch.randn(
#     B,
#     total_tokens,
#     HIDDEN_DIM
# )

# spatial_shapes = torch.tensor([
#     [8,8],
#     [4,4],
#     [2,2]
# ], dtype=torch.long)

# level_start_index = torch.tensor([
#     0,
#     64,
#     80
# ], dtype=torch.long)

# reference_points = get_reference_points(
#     spatial_shapes=spatial_shapes,
#     batch_size=B,
#     device= value.device
# )

# output = model(
#     query,
#     reference_points,
#     value,
#     spatial_shapes,
#     level_start_index
# )

# print("Output:", output)