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
        label_name = (os.path.splitext(image_name)[0] + '.txt')
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
                        if len(parts) != 5:
                             continue
                        
                        cls_id = int(parts[0])

                        #YOLO format: normalized [x_center, y_center, width, height]
                        x_c, y_c, w,h = map(float, parts[1:])

                        #Convert to absolute [x1, y1, x2, y2] for DINO/RT-DETR
                        x1 = (x_c - w/2) * origin_w
                        y1 = (y_c - h/2) * origin_h
                        x2 = (x_c + w/2) * origin_w
                        y2 = (y_c + h/2) * origin_h

                        #Clamp coordinates to valid image boundaries
                        x1 = max(0.0, min(x1,float(origin_w)))
                        y1 = max(0.0, min(y1,float(origin_h)))

                        x2 = max(0.0, min(x2,float(origin_w)))
                        y2 = max(0.0, min(y2,float(origin_h)))

                        #Skip ivalid boxes
                        if x2 <= x1 or y2 <= y1:
                            continue

                        boxes.append([x1,y1,x2,y2])
                        labels.append(cls_id)

        #Handle empty images (no bounding boxes)
        # if len(boxes) == 0:
        #      boxes = torch.zeros((0,4), dtype=torch.float32)
        #      labels = torch.zeros((0,), dtype=torch.int64)
        # else:
        #      boxes = torch.tensor(boxes, dtype=torch.float32)
        #      labels = torch.tensor(labels, dtype=torch.int64)

        # #Create target dictionary structure
        # target = {
        #      "boxes": boxes,
        #      "labels": labels,
        #      "image_id": torch.tensor(index),
        #      "orig_size": torch.tensor([origin_h,origin_w]),
        #      "size": torch.tensor([origin_h,origin_w])
        # }

       
    
        #Apply transform Albumentation
        if self.transform:
             #Albumentation
            augmented = self.transform(image=image, 
                                        bboxes = boxes, 
                                        labels = labels)
            image = augmented["image"]
            #Convert them to tensors
            augmented_boxes = augmented["bboxes"]
            augmented_labels = augmented["labels"]

        else:
            augmented_boxes = boxes
            augmented_labels = labels

            image = torch.from_numpy(
                image
            ).permute(
                2,0,1
            ).float() / 255.0

         #Get transformed size
        _,new_h,new_w = image.shape
        #Convert to tensors
        if len(augmented_boxes) > 0:

            boxes_tensor = torch.tensor(
                augmented_boxes,
                dtype=torch.float32
            ).reshape(-1,4)

            #Pascal VOC
            x1 = boxes_tensor[:,0]
            y1 = boxes_tensor[:,1]
            x2 = boxes_tensor[:,2]
            y2 = boxes_tensor[:,3]

            #Convert to cxcywh
            cx = (x1+x2) / 2
            cy = (y1+y2) / 2
            w = x2-x1
            h = y2-y1

            boxes_tensor = torch.stack(
                 [cx,cy,w,h],
                 dim=-1
            )

            #Normalize
            boxes_tensor[:,[0,2]] /=new_w
            boxes_tensor[:,[1,3]] /=new_h

            labels_tensor = torch.tensor(
                 augmented_labels,
                 dtype=torch.int64
            )

        else:
            boxes_tensor = torch.zeros(
                (0,4),
                dtype=torch.float32
            )

            labels_tensor = torch.zeros(
                 (0,),
                 dtype=torch.int64
            )

        
        target = {
            "boxes":boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(
                index
            ),
            "orig_size": torch.tensor(
                 [origin_h,origin_w]
            ),
            "size": torch.tensor(
                 [new_h,new_w]
            )
        }

        if boxes_tensor.numel() > 0:
            assert boxes_tensor.min() >= 0.0
            assert boxes_tensor.max() <= 1.0

                                        
        return image, target

#RT DETR like models can't accept batch images of varying dimensions so we intilaize a collate function
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    #Pad images or stack if transforms already resized them to static dims
    images = torch.stack(images,dim=0)
    return images,targets
    
            
