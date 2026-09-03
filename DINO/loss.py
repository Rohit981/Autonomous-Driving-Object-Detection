import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import generalized_box_iou
from matcher import HungarianMatcher

class DINOLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 matcher,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 weight_class=1.0,
                 weight_bbox=5.0,
                 weight_giou=2.0):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou


    #Box Conversion

    @staticmethod
    def box_cxcywh_to_xyxy(boxes):

        cx,cy,w,h = boxes.unbind(
            dim=-1
        )

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h

        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return torch.stack(
            [x1,
            y1,
            x2,
            y2
            ],
            dim=-1
        )

    #Sigmoid Focal Loss
    def sigmoid_focal_loss(
            self,
            inputs,
            targets
    ):
        prob = inputs.sigmoid()

        ce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none"
        )

        p_t = (
            prob * targets
            +
            (1-prob) * (1-targets)
        )

        loss = ce_loss * (
            (1-p_t)
            ** self.focal_gamma
        )

        alpha_t = (
            self.focal_alpha * targets
            +
            (1 - self.focal_alpha)
            *(1 - targets)
        )

        loss = alpha_t * loss

        return loss

    #Classification Loss
    def loss_labels(
            self,
            class_logits,
            targets,
            indices
    ):
        batch_size, num_queries, num_classes = (
            class_logits.shape
        )

        #Intially every query is background / negative
        target_classes = torch.zeros(
            (batch_size,
            num_queries,
            num_classes),
            dtype=torch.float32,
            device=class_logits.device
        )

        #Assign positive class to match queries
        for batch_idx, (
            pred_indices,
            target_indices
        ) in enumerate(indices):

            if pred_indices.numel() == 0:
                continue

            target_labels = targets[
                batch_idx
            ]["labels"]

            matched_labels = target_labels[
                target_indices
            ]

            target_classes[
                batch_idx,
                pred_indices,
                matched_labels
            ] = 1.0

        loss = self.sigmoid_focal_loss(
            class_logits,
            target_classes
        )

        #Average over number of matched targets
        num_boxes = sum(
            len(target["labels"])
            for target in targets
        )

        num_boxes = max(
            num_boxes,
            1
        )

        loss = loss.sum() / num_boxes

        return loss

    #Bounding Box L1 Loss
    def loss_boxes(
            self,
            pred_boxes,
            targets,
            indices
    ):
        src_boxes = []
        target_boxes = []

        for batch_idx, (
            pred_indices,
            target_indices
        ) in enumerate(indices):

            if pred_indices.numel() == 0:
                continue

            src_boxes.append(
                pred_boxes[
                    batch_idx,
                    pred_indices
                ]
            )

            target_boxes.append(
                targets[
                    batch_idx
                ]["boxes"][
                    target_indices
                ]
            )

            #Handle batch containing no objects
            if len(src_boxes) == 0:
                zero = torch.tensor(
                    0.0,
                    device=pred_boxes.device
                )

                return zero, zero

            src_boxes = torch.cat(
                src_boxes,
                dim=0
            )

            target_boxes = torch.cat(
                target_boxes,
                dim=0
            )

            #L1 Loss
            loss_bbox = F.l1_loss(
                src_boxes,
                target_boxes,
                reduction="none"
            ).sum()

            #GIOU Loss
            src_boxes_xyxy = (
                self.box_cxcywh_to_xyxy(
                    src_boxes
                )
            )

            target_boxes_xyxy = (
                self.box_cxcywh_to_xyxy(
                    target_boxes
                )
            )

            general_IOU = generalized_box_iou(
                src_boxes_xyxy,
                target_boxes_xyxy
            )

            #Only use corresponding matched pairs
            loss_giou = (
                1
                -
                torch.diag(
                    general_IOU
                )
            ).sum()

            num_boxes = max(
                sum(
                    len(target["boxes"])
                    for target in targets
                ),
                1
            )

            loss_bbox = (
                loss_bbox
                /
                num_boxes
            )

            loss_giou = (
                loss_giou
                /
                num_boxes
            )

            return loss_bbox, loss_giou

    def compute_detection_loss(
            self,
             class_logits,
            pred_boxes,
            targets
    ):
        #Hungarian Matcher
        indices = self.matcher(
            class_logits,
            pred_boxes,
            targets
        )

        #Classification Loss
        loss_class = self.loss_labels(
            class_logits,
            targets,
            indices
        )

        #Bounding Box loses
        loss_bbox, loss_giou = self.loss_boxes(
            pred_boxes,
            targets,
            indices
        )

        #Weighted total loss
        total_loss = (
            self.weight_class
            * loss_class

            +self.weight_bbox
            * loss_bbox

            +self.weight_giou
            *loss_giou
        )

        return {
            "loss_total": total_loss,

            "loss_class": loss_class,

            "loss_bbox": loss_bbox,

            "loss_giou": loss_giou,

            "indices": indices
        }

    #Main Forward
    def forward(
            self,
            class_logits,
            pred_boxes,
            targets,
            auxiliary_class_logits=None,
            auxiliary_boxes=None
    ):
        #Main / Final Decoder layer loss

        main_losses = self.compute_detection_loss(
            class_logits,
            pred_boxes,
            targets
        )

        total_loss = main_losses[
            "loss_total"
        ]

        #Auxiliary losses
        auxiliary_losses = []

        if(
            auxiliary_class_logits is not None
            and auxiliary_boxes is not None
        ):
            for layer_id in range(len(auxiliary_class_logits)):

                aux_loss = self.compute_detection_loss(
                    auxiliary_class_logits[
                        layer_id
                    ],
                    auxiliary_boxes[
                        layer_id
                    ],
                    targets
                )

                auxiliary_losses.append(aux_loss)

                total_loss = (
                    total_loss
                    +
                    aux_loss["loss_total"]
                )

        #Return Loss Dictionary
        return {
            #Total including auxiliary losses
            "loss_total": total_loss,

            #Main losses
            "loss_class": main_losses["loss_class"],

            "loss_bbox": main_losses["loss_bbox"],
            
            "loss_giou": main_losses["loss_giou"],

            "indices": main_losses["indices"],

            "auxiliary_losses": auxiliary_losses
        }

       


#Test
B= 2
NUM_QUERIES = 10
NUM_CLASSES = 10
NUM_LAYERS = 3
NUM_AUXILIARY_LAYERS = NUM_LAYERS - 1

auxiliary_class_logits = [
    torch.rand(
        B,
        NUM_QUERIES,
        NUM_CLASSES
    )
    for _ in range(NUM_AUXILIARY_LAYERS)
]

auxiliary_boxes = [
    torch.rand(
        B,
        NUM_QUERIES,
        4
    )
    for _ in range(NUM_AUXILIARY_LAYERS)
]

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

hungarian_matcher = HungarianMatcher()

dino_loss = DINOLoss(
    num_classes=NUM_CLASSES,
    matcher=hungarian_matcher)

losses = dino_loss(
    class_logits,
    pred_boxes,
    targets,
    auxiliary_class_logits,
    auxiliary_boxes
)

total_loss = losses["loss_total"]
loss_class = losses["loss_class"]
loss_bbox = losses["loss_bbox"]
loss_giou = losses["loss_giou"]
indices = losses["indices"]
auxiliary_loss = losses["auxiliary_losses"]


print("Total Loss:", total_loss)
print("Loss Class:", loss_class)
print("Loss Bbox:", loss_bbox)
print("Loss Giou:", loss_giou)
print("Indices:", indices)
print("auxiliary_loss:", auxiliary_loss)

for layer_id, aux_loss in enumerate(
    losses["auxiliary_losses"]
):

    print(
        f"\nAuxiliary Layer {layer_id}"
    )

    print(
        "Total:",
        aux_loss["loss_total"]
    )

    print(
        "Class:",
        aux_loss["loss_class"]
    )

    print(
        "BBox:",
        aux_loss["loss_bbox"]
    )

    print(
        "GIoU:",
        aux_loss["loss_giou"]
    )
