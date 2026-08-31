import os
import cv2
import torch
from torch.utils.data import Dataset


class BDD100kDataset(Dataset):
    def __init__(self,
                 image_dir,
                 labels_dir,
                 classes_names,
                 transform=None
        ):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.class_names = classes_names
        self.transform = transform

        #Get all image files
        self.image_files = sorted(
                [
                    file for file in os.listdir(self.image_dir)
                    if file.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
        )

        if len(self.image_files) == 0:
                raise RuntimeError(
                    f"No Image file found in {self.image_dir}"
                )
        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        #Load Image
        image_name = self.image_files[index]
        image_path = os.path.join(
            self.image_dir,
            image_name
        )
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Couldn't find the image {image_path}"
            )
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        origin_h, origin_w, _ = image.shape


        #Load XML annotation
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(
                self.labels_dir,
                label_name
        )

        
        boxes = []
        labels = []

        #Extract Objects
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                             cls_id = int(parts[0])

                             #YOLO format: normalized [x_center, y_center, width, height]
                             x_c, y_c, w,h = map(float, parts[1:])

                             #Convert to absolute [x1, y1, x2, y2] for DINO/RT-DETR
                             x1 = (x_c - w/2) * origin_w
                             y1 = (y_c - h/2) * origin_h
                             x2 = (x_c + w/2) * origin_w
                             y2 = (y_c + h/2) * origin_h

                             boxes.append([x1,y1,x2,y2])
                             labels.append(cls_id)

        #Handle empty images (no bounding boxes)
        if len(boxes) == 0:
             boxes = torch.zeros((0,4), dtype=torch.float32)
             labels = torch.zeros((0,), dtype=torch.int64)
        else:
             boxes = torch.tensor(boxes, dtype=torch.float32)
             labels = torch.tensor(labels, dtype=torch.int64)

        #Create target dictionary structure
        target = {
             "boxes": boxes,
             "labels": labels,
             "image_id": torch.tensor(index),
             "orig_size": torch.tensor([origin_h,origin_w]),
             "size": torch.tensor([origin_h,origin_w])
        }

        #Apply transform Albumentation
        if self.transform:
             #Albumentation
            augmented = self.transform(image=image, 
                                        bboxes = target["boxes"], 
                                        labels = target["labels"])
            image = augmented["image"]
            #Convert them to tensors
            target["boxes"] = torch.tensor(augmented["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(augmented["labels"], dtype=torch.int64)
                                        
        return image, target

    #RT DETR like models can't accept batch images of varying dimensions so we intilaize a collate function
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        #Pad images or stack if transforms already resized them to static dims
        images = torch.stack(images,dim=0)
        return images,targets
        
            
