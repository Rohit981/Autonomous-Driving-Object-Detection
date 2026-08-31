import torch.nn as nn
from backbone import DINOBackbone

class DINO(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = DINOBackbone()
        self.input_proj = ...
        self.positional_encoding = ...
        self.encoder = ...
        self.decoder = ...
        self.query_selection = ...
        self.class_head = ...
        self.bbox_head = ...

    def forward(self, images):
        #Backbone
        features = self.backbone(images)

        #Projection feature levels
        projection = self.input_proj(features)

        #Positional encodings
        pos = ...

        #Transformer Encoder
        memory = self.encoder(
            features,
            pos
        )

        #Select Initial queries
        queries, reference_points = \
            self.query_selection(memory)

        #Decoder
        hs,references = self.decoder(
            queries,
            reference_points,
            memory
        )

        #Predictions
        output_class = ...
        output_boxes = ...

        return {
            "pred_logits": output_class,
            "pred_boxes": output_boxes
        }