import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Deformable_Attention import MultiScaleDeformableAttention
from reference_embedding import ReferencePointEmbedding

class DecoderSelfAttention(nn.Module):
    def __init__(self,
                 hidden_dims=256,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self,
                query,
                query_pos=None,
                attn_mask=None):

        #Add Positional information for Q and K
        if query_pos is not None:
            q = query + query_pos
            k = query + query_pos

        else:
            q = query
            k = query

        v = query

        output,_ = self.attention(
            q,
            k,
            v,
            attn_mask=attn_mask
        )

        return output

class DeformableDecoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()

        #Create Reference Point Embedding
        self.reference_point_embedding = ReferencePointEmbedding(
            hidden_dim
        )

        #SELF ATTENTION
        self.self_attn = DecoderSelfAttention(
            hidden_dims=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        #Apply first layer normalization and dropout
        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        #Deformable Cross Attention
        self.cross_atn = MultiScaleDeformableAttention(
            hidden_dims=hidden_dim,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        #Feed Forward Network
        self.linear1 = nn.Linear(
            hidden_dim,
            dim_feedforward
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            hidden_dim
        )

        self.activation = nn.ReLU()

        self.dropout3 = nn.Dropout(
            dropout
        )

        self.norm3 = nn.LayerNorm(
            hidden_dim
        )

    def forward(self,
                query,
                reference_points,
                memory,
                pos,
                spatial_shapes,
                level_start_index):

        #Self Attention
        query_pos = self.reference_point_embedding(
            reference_points
        )

        self.attn_output = self.self_attn(
            query=query,
            query_pos=query_pos
        )

        query = query + self.dropout1(
            self.attn_output
        )

        query = self.norm1(
            query
        )

        #Deformable Cross Attention
        reference_points_for_attention = (
            reference_points
            .unsqueeze(2)
            .repeat(
                1,
                1,
                spatial_shapes.shape[0],
                1
            )
        )

        cross_atn_output = self.cross_atn(
            query = query + query_pos,
            reference_points = reference_points_for_attention,
            value = memory,
            spatial_shapes = spatial_shapes,
            level_start_index=level_start_index
        )

        query = query + self.dropout2(
            cross_atn_output
        )

        query = self.norm2(
            query
        )

        #Feed Forward Network
        ff_output = self.linear2(
            self.dropout3(
                self.activation(
                    self.linear1(query)
                )
            )
        )

        query = query + ff_output

        query = self.norm3(
            query
        )

        return query

#Decoder Test
# B = 1
# NUM_QUERIES = 10
# HIDDEN_DIM = 64
# NUM_LEVELS = 3
# NUM_HEADS = 4
# NUM_POINTS = 2

# spatial_shapes = torch.tensor([
#     [8,8],
#     [4,4],
#     [2,2]
# ], dtype=torch.long)

# total_tokens = 84

# level_start_index = torch.tensor([
#     0,
#     64,
#     80
# ], dtype=torch.long)

# memory = torch.randn(
#     B,
#     total_tokens,
#     HIDDEN_DIM
# )

# pos = torch.randn(
#     B,
#     total_tokens,
#     HIDDEN_DIM
# )

# query = torch.randn(
#     B,
#     NUM_QUERIES,
#     HIDDEN_DIM
# )

# reference_points = torch.rand(
#     B,
#     NUM_QUERIES,
#     2
# )

# level_start_index = torch.tensor([
#     0,
#     64,
#     80
# ])

# decoder_layer = DeformableDecoderLayer(
#     hidden_dim=HIDDEN_DIM,
#     num_heads=NUM_HEADS,
#     num_levels=NUM_LEVELS,
#     num_points=NUM_POINTS
# )

# output = decoder_layer(
#     query,
#     reference_points,
#     memory,
#     pos,
#     spatial_shapes,
#     level_start_index
# )

# print("Decoder Output Shape:", output.shape)
