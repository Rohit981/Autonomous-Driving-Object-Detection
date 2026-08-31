import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGE_SIZE = 640

#Train Transform
def get_train_transforms():

    return A.Compose(
        [
            #Resize while preserving aspect ratio

            A.LongestMaxSize(
                max_size=IMAGE_SIZE
            ),

            #Pad to fixed size
            A.PadIfNeeded(
                min_height=IMAGE_SIZE,
                min_width=IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),

            #Geometric Augmentation
            A.HorizontalFlip(
                p=0.5
            ),

            A.Affine(
                scale=(0.9,1.1),
                translate_percent=(-0.1,0.1),
                rotate=(-10,10),
                shear=(-5,5),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.3
            ),

            #Appearence Augmentation
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1
                    ),
                ],
                p=0.4
            ),

            #Slight Blur
            A.OneOf(
                [
                    A.Blur(
                        blur_limit=3
                    ),
                    A.GaussianBlur(
                        blur_limit=(3,5)
                    ),
                ],
                p=0.1
            ),

            #Normalize
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std= (0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),

            #Numpy to Pytorch Tensor
            ToTensorV2(),
        ],

        bbox_params= A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            #Remove boxes that become too small
            min_area=1.0,
            min_visibility=0.1,
        ),   
    )

#Val Transforms
def get_val_transforms():
    return A.Compose(
        [
            #No random during validation
            A.LongestMaxSize(
                max_size=IMAGE_SIZE
            ),

            A.PadIfNeeded(
                min_height=IMAGE_SIZE,
                min_width=IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],

        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.1
        )
    )