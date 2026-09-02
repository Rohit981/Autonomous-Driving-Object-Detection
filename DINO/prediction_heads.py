import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 num_classes=10):
        super().__init__()

        self.class_head = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self,x):
        return self.class_head(x)

class BoundingBoxHead(nn.Module):
    def __init__(self,
                 hidden_dim=256):
        super().__init__()

        self.bbox_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                4
            )
        )

    def forward(self,x):
        return self.bbox_head(x)

#TEST
B = 2
NUM_QUERIES = 10
HIDDEN_DIM = 64
NUM_CLASSES = 10

decoder_features = torch.randn(
    B,
    NUM_QUERIES,
    HIDDEN_DIM
)

class_head = ClassificationHead(
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
)

bbox_head = BoundingBoxHead(
    hidden_dim=HIDDEN_DIM
)

class_logits = class_head(
    decoder_features
)

bbox_output = bbox_head(
    decoder_features
)

print("Class Logits:", class_logits.shape)
print("BBOX OUTPUT:", bbox_output.shape)