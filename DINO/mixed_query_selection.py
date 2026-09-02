import torch
import torch.nn as nn

class MixedQuerySelection(nn.Module):
    def __init__(self,
                 hidden_dims=256,
                 num_classes=10,
                 num_queries=300):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dims

        #Learned content queries
        #These are the decoder target/content embeddings

        self.query_content = nn.Embedding(
            num_queries,
            hidden_dims
        )

        #Encoder Classification scores
        self.class_head = nn.Linear(
            hidden_dims,
            num_classes
        )

        #Generate Initial reference boxes
        self.bbox_head = nn.Linear(
            hidden_dims,
            4
        )

    def forward(self, memory):

        batch_size, total_tokens, _ = memory.shape

        #Class prediction from encoder features
        class_logits = self.class_head(
            memory
        )

        # [B, total_tokens, num_classes]

        #Calculate objectness score

        scores = class_logits.max(
            dim=-1
        ).values

        # [B, total_tokens]
        #Select top-K encoder features

        _, topk_indices = torch.topk(
            scores,
            k = self.num_queries,
            dim=1
        )

        #topk_indices:
        #[B, num_queries]
        #Predict encoder reference boxes

        all_reference_boxes = self.bbox_head(
            memory
        ).sigmoid()

        #Select TOP-K reference boxes

        reference_boxes = torch.gather(
            all_reference_boxes,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(
                -1,
                -1,
                4
            )
        )

        # [B, num_queries, hidden_dim]
        #Generate decoder content queries

        target = self.query_content.weight.unsqueeze(
            0
        ).expand(
            batch_size,
            -1,
            -1
        )

        # [B, num_queries,4]
        return (
            target,
            reference_boxes,
            topk_indices
        )
    
#Test
B=2
TOTAL_TOKENS=84
HIDDEN_DIM=64
NUM_CLASSES=10
NUM_QUERIES=10

memory = torch.randn(
    B,
    TOTAL_TOKENS,
    HIDDEN_DIM
)

model = MixedQuerySelection(
    hidden_dims=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    num_queries=NUM_QUERIES
)

target, reference_boxes,topk_indices = model(
    memory
)

print("Target:", target.shape)
print("Reference Boxes:", reference_boxes.shape)
print("TOPK Indices:", topk_indices.shape)
