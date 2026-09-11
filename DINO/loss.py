import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import generalized_box_iou
from .matcher import HungarianMatcher
from Utils import box_cxcywh_to_xyxy

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
            box_cxcywh_to_xyxy(
                src_boxes
            )
        )

        target_boxes_xyxy = (
            box_cxcywh_to_xyxy(
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

    #Compute DN loss
    def compute_dn_loss(
            self,
            dn_class_logits,
            dn_boxes,
            targets,
            dn_meta
    ):
        batch_size = len(targets)

        total_class_loss = 0.0
        total_bbox_loss = 0.0
        total_giou_loss = 0.0

        num_valid_images = 0

        target_indices_list = dn_meta['target_indices']

        for batch_idx in range(batch_size):

            target_indices = target_indices_list[
                batch_idx
            ]

            if len(target_indices) == 0:
                continue

            target_indices = torch.as_tensor(
                target_indices,
                device=dn_class_logits.device,
                dtype=torch.long
            )

            #Select DN predictions corresponding to the original targets
            pred_logits = dn_class_logits[
                batch_idx,
                target_indices
            ]

            pred_boxes = dn_boxes[
                batch_idx,
                target_indices
            ]

            target_labels = targets[
                batch_idx
            ]["labels"]

            target_boxes = targets[
                batch_idx
            ]['boxes']

            num_dn_groups = dn_meta[
                "num_dn_groups"
            ]

            #Repeat targets for every DN group
            target_labels = target_labels.repeat(
                num_dn_groups
            )
            target_boxes = target_boxes.repeat(
                num_dn_groups,
                1
            )

            #Negative DN targets
            negative_target_labels = target_labels.clone()
            negative_target_boxes = target_boxes.clone()

            #Combine
            target_labels = torch.cat(
                [
                    target_labels,
                    negative_target_labels
                ],
                dim=0
            )

            target_boxes = torch.cat(
                [
                    target_boxes,
                    negative_target_boxes
                ],
                dim=0
            )

            # print(
            #     "Pred logits:",
            #     pred_logits.shape
            # )

            # print(
            #     "Target labels:",
            #     target_labels.shape
            # )

            # print(
            #     "Pred boxes:",
            #     pred_boxes.shape
            # )

            # print(
            #     "Target boxes:",
            #     target_boxes.shape
            # )

            #Classification
            class_loss = F.cross_entropy(
                pred_logits,
                target_labels
            )

            #L1 Box loss
            bbox_loss = F.l1_loss(
                pred_boxes,
                target_boxes
            )

            #GIOU loss
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

            giou = generalized_box_iou(
                pred_boxes_xyxy,
                target_boxes_xyxy
            )

            giou_loss = (
                1 - torch.diag(giou)
            ).mean()

            total_class_loss += class_loss
            total_bbox_loss += bbox_loss
            total_giou_loss += giou_loss

            num_valid_images +=1

        #Prevent division by zero
        if num_valid_images == 0:
            zero_loss = torch.tensor(
                0.0,
                device=dn_class_logits.device
            )

            return {
                "loss_total": zero_loss,
                "loss_class": zero_loss,
                "loss_bbox": zero_loss,
                "loss_giou": zero_loss
            }

        #Average across images
        total_class_loss = (
            total_class_loss / num_valid_images
        )

        total_bbox_loss = (
            total_bbox_loss / num_valid_images
        )

        total_giou_loss = (
            total_giou_loss / num_valid_images
        )

        #Weight DN loss
        total_loss = (
            self.weight_class * total_class_loss
            +
            self.weight_bbox * total_bbox_loss
            +
            self.weight_giou * total_giou_loss
        )

        return {
            "loss_total": total_loss,
            "loss_class": total_class_loss,
            "loss_bbox": total_bbox_loss,
            "loss_giou": total_giou_loss
        }
         

    #Main Forward
    def forward(
            self,
            class_logits,
            pred_boxes,
            targets,
            auxiliary_class_logits=None,
            auxiliary_boxes=None,
            dn_class_logits=None,
            dn_boxes=None,
            dn_meta=None
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

        #DN losses
        dn_losses = []

        if (
            dn_class_logits is not None
            and dn_boxes is not None
            and dn_meta is not None
        ):
            for layer_id in range(len(dn_class_logits)):
                layer_dn_losses = self.compute_dn_loss(
                    dn_class_logits[layer_id],
                    dn_boxes[layer_id],
                    targets,
                    dn_meta
                )

                dn_losses.append(layer_dn_losses)

                total_loss = (
                    total_loss
                    +
                    layer_dn_losses["loss_total"]
                )

        #Return Loss Dictionary
        return {
            #Total including auxiliary losses
            "loss_total": total_loss,

            #Main loss total
            "main_loss_total": main_losses["loss_total"],

            #Main losses
            "loss_class": main_losses["loss_class"],

            "loss_bbox": main_losses["loss_bbox"],
            
            "loss_giou": main_losses["loss_giou"],

            "indices": main_losses["indices"],

            "auxiliary_losses": auxiliary_losses,

            "dn_losses": dn_losses
        }

#Test
# B= 2
# NUM_QUERIES = 10
# NUM_CLASSES = 10
# NUM_LAYERS = 3
# NUM_AUXILIARY_LAYERS = NUM_LAYERS - 1

# auxiliary_class_logits = [
#     torch.rand(
#         B,
#         NUM_QUERIES,
#         NUM_CLASSES
#     )
#     for _ in range(NUM_AUXILIARY_LAYERS)
# ]

# auxiliary_boxes = [
#     torch.rand(
#         B,
#         NUM_QUERIES,
#         4
#     )
#     for _ in range(NUM_AUXILIARY_LAYERS)
# ]

# class_logits = torch.randn(
#     B,
#     NUM_QUERIES,
#     NUM_CLASSES
# )

# pred_boxes = torch.rand(
#     B,
#     NUM_QUERIES,
#     4
# )

# targets = [
#     {
#         "labels": torch.tensor(
#             [1,3,5]
#         ),
#         "boxes": torch.rand(
#             3,
#             4
#         )
#     },
#     {
#         "labels": torch.tensor(
#             [2,4]
#         ),

#         "boxes": torch.rand(
#             2,
#             4
#         )
#     }
# ]

# hungarian_matcher = HungarianMatcher()

# dino_loss = DINOLoss(
#     num_classes=NUM_CLASSES,
#     matcher=hungarian_matcher)

# losses = dino_loss(
#     class_logits,
#     pred_boxes,
#     targets,
#     auxiliary_class_logits,
#     auxiliary_boxes
# )

# total_loss = losses["loss_total"]
# loss_class = losses["loss_class"]
# loss_bbox = losses["loss_bbox"]
# loss_giou = losses["loss_giou"]
# indices = losses["indices"]
# auxiliary_loss = losses["auxiliary_losses"]


# print("Total Loss:", total_loss)
# print("Loss Class:", loss_class)
# print("Loss Bbox:", loss_bbox)
# print("Loss Giou:", loss_giou)
# print("Indices:", indices)
# print("auxiliary_loss:", auxiliary_loss)

# for layer_id, aux_loss in enumerate(
#     losses["auxiliary_losses"]
# ):
#     print(
#         f"\nAuxiliary Layer {layer_id}"
#     )

#     print(
#         "Total:",
#         aux_loss["loss_total"]
#     )

#     print(
#         "Class:",
#         aux_loss["loss_class"]
#     )

#     print(
#         "BBox:",
#         aux_loss["loss_bbox"]
#     )

#     print(
#         "GIoU:",
#         aux_loss["loss_giou"]
#     )
