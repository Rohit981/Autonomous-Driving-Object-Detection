import torch
import torch.nn as nn

class MixedQuerySelection(nn.Module):
    def __init__(self,
                 hidden_dims=256,
                 num_classes=10,
                 num_queries=300):
        super().__init__()

        self.num_queries = num_queries

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

        #Transform selected encoder features into decoder content queries
        self.query_projection = nn.Linear(
            hidden_dims,
            hidden_dims
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

        topk_scores, topk_indices = torch.topk(
            scores,
            k = self.num_queries,
            dim=1
        )

        #topk_indices:
        #[B, num_queries]

        #Select corresponding encoder features

        query_features = torch.gather(
            memory,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(
                -1,
                -1,
                memory.shape[-1]
            )
        )

        # [B, num_queries, hidden_dim]
        #Generate decoder content queries

        target = self.query_projection(
            query_features
        )

        #Generate reference boxes
        bbox_predictions = self.bbox_head(
            memory
        )

        bbox_predictions = bbox_predictions.sigmoid()

        # [B, total_tokens, 4]
        reference_boxes = torch.gather(
            bbox_predictions,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(
                -1,
                -1,
                4
            )
        )

        # [B, num_queries,4]
        return (
            target,
            reference_boxes,
            topk_indices
        )
    
