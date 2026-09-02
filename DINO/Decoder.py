import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from Deformable_Attention import MultiScaleDeformableAttention, get_reference_points
from reference_embedding import ReferencePointEmbedding
from prediction_heads import ClassificationHead, BoundingBoxHead

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
                reference_boxes,
                memory,
                spatial_shapes,
                level_start_index):

        #Self Attention
        query_pos = self.reference_point_embedding(
            reference_boxes
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
            reference_boxes
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

class DeformableDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers=6,
                 hidden_dim=256,
                 num_classes=10):
        super().__init__()

        self.num_layers = num_layers

        #Decoder Layers
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer)
            for _ in range(num_layers)
        ])

        #Classification Heads One for each decoder layer
        self.class_heads = nn.ModuleList([
            ClassificationHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
            for _ in range(num_layers)
        ])

        #Bounding box heads One for each decoder layer
        self.bbox_heads = nn.ModuleList([
            BoundingBoxHead(
                hidden_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])

    def forward(self,
                query,
                reference_boxes,
                memory,
                spatial_shapes,
                level_start_index):

        intermediate_outputs = []
        intermediate_class_logits = []
        intermediate_boxes = []

        #Process each decoder layer
        for layer_id, layer in enumerate(self.layers):

            #Run Decoder layer
            query = layer(
                query=query,
                reference_boxes= reference_boxes,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )

            #Auxiliary classification prediction
            class_logits = self.class_heads[layer_id](
                query
            )

            #Bounding box prediction
            bbox_delta = self.bbox_heads[layer_id](
                query
            )

            #Store decoder features
            intermediate_boxes.append(bbox_delta)
            intermediate_outputs.append(query)
            intermediate_class_logits.append(class_logits)

        return(
            intermediate_outputs,
            intermediate_class_logits,
            intermediate_boxes
        )

#Decoder Test
B = 1
NUM_QUERIES = 10
HIDDEN_DIM = 64
NUM_LAYERS = 3
NUM_CLASSES = 10
NUM_HEADS = 4
NUM_LEVELS = 3
NUM_POINTS = 2
TOTAL_TOKENS = 84

decoder_layer = DeformableDecoderLayer(
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_levels=NUM_LEVELS,
    num_points=NUM_POINTS
)

decoder = DeformableDecoder(
    decoder_layer=decoder_layer,
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
)

query = torch.randn(
    B,
    NUM_QUERIES,
    HIDDEN_DIM
)


memory = torch.randn(
    B,
    TOTAL_TOKENS,
    HIDDEN_DIM
)

spatial_shapes = torch.tensor([
    [8, 8],
    [4, 4],
    [2, 2]
], dtype=torch.long)

level_start_index = torch.tensor([
    0,
    64,
    80
], dtype=torch.long)

reference_boxes = torch.rand(
    B,
    NUM_QUERIES,
    4
)

output, class_logits, box_outputs = decoder(
        query,
        reference_boxes,
        memory,
        spatial_shapes,
        level_start_index
)

print(len(output))
print(len(class_logits))
print(len(box_outputs))

for i in range(NUM_LAYERS):

    print(f"\nLayer {i+1}")

    print(
        "Decoder Output:",
        output[i].shape
    )

    print(
        "Class Logits:",
        class_logits[i].shape
    )

    print(
        "Box Output:",
        box_outputs[i].shape
    )
