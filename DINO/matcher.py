import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self,
                 cost_class=1.0,
                 cost_bbox=5.0,
                 cost_giou=2.0):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        if (
            cost_class==0
            and cost_bbox==0
            and cost_giou==0
        ):
            raise ValueError(
                "At least one matching cost must be non-zero"
            )

    @torch.no_grad()
    def forward(
        self,
        class_logits,
        pred_boxes,
        targets
    ):
        batch_size, num_queries, num_classes =(
            class_logits.shape
        )

        indices = []

        for batch_idx in range(batch_size):

            #Prediction for current image
            pred_logits = class_logits[
                batch_idx
            ]

            pred_bbox = pred_boxes[
                batch_idx
            ]

            #Ground Truth
            target_labels = targets[
                batch_idx
            ]["labels"]

            target_boxes = targets[
                batch_idx
            ]["boxes"]

            #Handle images with no objects
            if len(target_labels) == 0:

                empty_prediction_indices = torch.empty(
                    0,
                    dtype=torch.int64,
                    device=pred_boxes.device
                )

                empty_target_indices = torch.empty(
                    0,
                    dtype=torch.int64,
                    device=pred_boxes.device
                )

                indices.append(
                    (
                        empty_prediction_indices,
                        empty_target_indices
                    )
                )

                continue

            #Classification Cost
            pred_prob = pred_logits.softmax(
                dim=-1
            )

            cost_class = -pred_prob[
                :,
                target_labels
            ]

            #L1 bounding box Cost
            cost_bbox = torch.cdist(
                pred_bbox,
                target_boxes,
                p=1
            )

            #GIOU Cost
            pred_boxes_xyxy = self.box_cxcywh_to_xyxy(
                pred_bbox
            )

            target_boxes_xyxy = self.box_cxcywh_to_xyxy(
                target_boxes
            )

            cost_giou = -generalized_box_iou(
                pred_boxes_xyxy,
                target_boxes_xyxy
            )

            #Final Matching Cost
            cost_matrix = (
                self.cost_class * cost_class
                +
                self.cost_bbox * cost_bbox
                +
                self.cost_giou * cost_giou
            )

            #Hungarian Matching

            cost_matrix = cost_matrix.cpu()

            prediction_indices, target_indices = (
                linear_sum_assignment(
                    cost_matrix
                )
            )

            prediction_indices = torch.as_tensor(
                prediction_indices,
                dtype=torch.int64,
                device=pred_boxes.device
            )

            target_indices = torch.as_tensor(
                target_indices,
                dtype=torch.int64,
                device=pred_boxes.device
            )

            indices.append(
                (
                    prediction_indices,
                    target_indices
                )
            )

        return indices

    @staticmethod
    def box_cxcywh_to_xyxy(
        boxes
    ):
        cx,cy,w,h = boxes.unbind(
            dim=-1
        )

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h

        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return torch.stack(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dim=-1
        )

#Test
B= 2
NUM_QUERIES = 10
NUM_CLASSES = 10

class_logits = torch.randn(
    B,
    NUM_QUERIES,
    NUM_CLASSES
)

pred_boxes = torch.rand(
    B,
    NUM_QUERIES,
    4
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

matcher = HungarianMatcher()

indices = matcher(
    class_logits,
    pred_boxes,
    targets
)

for batch_idx, (
    prediction_indices,
    target_indices
) in enumerate(indices):

    print(
        f"\nBatch {batch_idx}"
    )

    print(
        "Matched predictions:",
        prediction_indices
    )

    print(
        "Matched targets:",
        target_indices
    )

