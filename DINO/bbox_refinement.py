import torch
import torch.nn as nn

class BBoxRefinement(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def inverse_sigmoid(
        x,
        eps=1e-6
    ):
        x = x.clamp(
            min=eps,
            max = 1-eps
        )

        return torch.log(
            x / (1-x)
        )

    def forward(
            self,
            reference_boxes,
            bbox_delta
    ):
        #reference_boxes:
        # [B,Q,4]
        # [cx,cy,w,h]
        #
        #bbox_delta:
        #[B,Q,4]

        reference_boxes_unactivated = (
            self.inverse_sigmoid(
                reference_boxes
            )
        )

        updated_boxes = (
            reference_boxes_unactivated
            +
            bbox_delta
        )

        updated_boxes = torch.sigmoid(
            updated_boxes
        )

        return updated_boxes

#Test
B = 1
NUM_QUERIES = 10

reference_boxes = torch.rand(
    B,
    NUM_QUERIES,
    4
)

bbox_delta = torch.randn(
    B,
    NUM_QUERIES,
    4
)

refinement = BBoxRefinement()

updated_boxes = refinement(
    reference_boxes,
    bbox_delta
)

print("Updated Boxes Shape:", updated_boxes.shape)
print("Updated Boxes min:", updated_boxes.min())
print("Updated Boxes max:", updated_boxes.max())


