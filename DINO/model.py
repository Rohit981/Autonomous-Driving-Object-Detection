import torch
import torch.nn as nn
from backbone import DINOBackbone
from Encoder import DeformableEncoder
from Decoder import DeformableDecoder,DeformableDecoderLayer
from mixed_query_selection import MixedQuerySelection
from feature_encoder import MultiScaleFeatureFlatten
from denoising import ContrastiveDenoising
from backbone import FeatureProjection
from Deformable_Attention import get_reference_points

class DINO(nn.Module):
    def __init__(self,
                 num_classes,
                 hidden_dim=256
                ):
        super().__init__()

        self.backbone = DINOBackbone()
        self.input_proj = nn.ModuleList([
            FeatureProjection(
                in_channels=256,
                hidden_dims=hidden_dim
            ),

            FeatureProjection(
                in_channels=512,
                hidden_dims=hidden_dim
            ),

            FeatureProjection(
                in_channels=1024,
                hidden_dims=hidden_dim
            )
        ])

        self.feature_encoder = MultiScaleFeatureFlatten(
            hidden_dim=hidden_dim
            )
        self.encoder = DeformableEncoder()
        self.decoder = DeformableDecoder(
            decoder_layer=DeformableDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=8,
                num_levels=3,
                num_points=4
            ),
            num_layers=6,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        self.query_selection = MixedQuerySelection(hidden_dims=hidden_dim,
                                                   num_classes=num_classes,
                                                   num_queries=300)
        self.denoising = ContrastiveDenoising(
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )

    def forward(self, 
                images,
                targets=None):
        #Backbone
        features = self.backbone(images)

        #Projection feature levels
        projected_features = [
            proj(feature)
            for proj, feature in zip(
                self.input_proj,
                features
            )
        ]

        #Multi-scale flattening + positional encoding
        (
            src_flatten,
            pos_flatten,
            spatial_shapes,
            level_start_index
        ) = self.feature_encoder(
            projected_features
        )

        # Generate encoder reference points
        reference_points = get_reference_points(
            spatial_shapes=spatial_shapes,
            batch_size=images.shape[0],
            device=images.device
        )

        #Deformable Encoder
        memory = self.encoder(
            src_flatten,
            pos_flatten,
            reference_points,
            spatial_shapes,
            level_start_index
        )
        
        #Select Initial queries
        query, reference_boxes, topk_indices = self.query_selection(
            memory,
            # spatial_shapes,
            # level_start_index
            )

        #Default DN Values
        dn_queries = None
        dn_boxes = None
        attn_mask = None
        dn_meta = None

        #DN only during training
        if self.training and targets is not None:
            dn_output = self.denoising(
                targets,
                images.device
            )

            dn_queries = dn_output["dn_queries"]
            dn_boxes = dn_output["dn_boxes"]
            attn_mask = dn_output["attn_mask"]
            dn_meta = dn_output["dn_meta"]

        #Decoder
        (
            intermediate_outputs,
            intermediate_class_logits,
            intermediate_boxes,
            intermediate_dn_class_logits,
            intermediate_dn_boxes
        ) = self.decoder(
            query = query,
            reference_boxes = reference_boxes,
            memory = memory,
            spatial_shapes = spatial_shapes,
            level_start_index = level_start_index,
            dn_queries=dn_queries,
            dn_boxes=dn_boxes,
            attn_mask=attn_mask
        )

        #Final decoder layer prediction
        output_class = intermediate_class_logits[-1]

        output_boxes = intermediate_boxes[-1]

        return {
            "pred_logits": output_class,
            "pred_boxes": output_boxes,

            "aux_class_logits":
                intermediate_class_logits[:-1],

            "aux_boxes":
                intermediate_boxes[:-1],

            "dn_class_logits":
                intermediate_dn_class_logits,

            "dn_boxes":
                intermediate_dn_boxes,

            "dn_meta":
                dn_meta
        }

#Test
NUM_CLASSES = 10
B = 2
C = 3
H = 256
W = 256

images = torch.randn(
    B,
    C,
    H,
    W
)

targets = [
    {
        "labels": torch.tensor(
            [1,3,5]
        ),
        "boxes": torch.rand(
            3,
            4
        )
    },
    {
        "labels": torch.tensor(
            [2,4]
        ),

        "boxes": torch.rand(
            2,
            4
        )
    }
]

model = DINO(
    num_classes=NUM_CLASSES
)

outputs = model(
    images,
    targets
)

print("Model training:", model.training)

print(
    "Pred logits:",
    outputs["pred_logits"].shape
)

print(
    "Pred boxes:",
    outputs["pred_boxes"].shape
)

print(
    "Aux logits:",
    len(outputs["aux_class_logits"])
)

print(
    "Aux boxes:",
    len(outputs["aux_boxes"])
)

print(
    "DN logits:",
    len(outputs["dn_class_logits"])
)

print(
    "DN boxes:",
    len(outputs["dn_boxes"])
)
