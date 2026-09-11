import matplotlib.pyplot as plt
import torch

  #Visualize Train and Test loss and Accuracy
def Visualize_loss_acc(train_loss,
                       val_loss, 
                       train_class_loss, 
                       val_class_loss, 
                       train_bbox_loss, 
                       val_bbox_loss, 
                       train_giou_loss, 
                       val_giou_loss):
    plt.figure(figsize=(12,5))
    epochs = range(1, len(train_loss) + 1)

    #Loss History
    plt.subplot(1,4,1)
    plt.plot(epochs,train_loss,label="Train Main Loss", color="blue")
    plt.plot(epochs,val_loss,label="Val Main loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Main Loss")
    plt.legend()
    plt.grid(True)

    #Loss Class History
    plt.subplot(1,4,2)
    plt.plot(epochs,train_class_loss,label="Train Class Loss", color="blue")
    plt.plot(epochs,val_class_loss,label="Val Class loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Class")
    plt.legend()
    plt.grid(True)

    #Loss BBOX History
    plt.subplot(1,4,3)
    plt.plot(epochs,train_bbox_loss,label="Train BBOX Loss", color="blue")
    plt.plot(epochs,val_bbox_loss,label="Val BBOX loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss BBOX")
    plt.legend()
    plt.grid(True)

    #Loss GIOU History
    plt.subplot(1,4,4)
    plt.plot(epochs,train_giou_loss,label="Train GIOU Loss", color="blue")
    plt.plot(epochs,val_giou_loss,label="Val GIOU loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss GIOU")
    plt.legend()
    plt.grid(True)

    plt.show()

#Convert normalized cxcywh boxes to normalized xyxy boxes
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

#Run DINO inference and return detections.
#Returns detections list of dictionaries, one per image

@torch.no_grad()
def get_detection(
   model,
   images,
   confidence_threshold=0.5
):
    model.eval()

    outputs = model(images)

    pred_logits = outputs["pred_logits"]

    print("Logits shape:", pred_logits.shape)
    print("Logits min:", pred_logits.min().item())
    print("Logits max:", pred_logits.max().item())

    print("Logits mean:", pred_logits.mean().item())
    print("Logits std:", pred_logits.std().item())


    pred_boxes = outputs["pred_boxes"]

    #[B,num_queries,num_classes]
    probabilities = pred_logits.sigmoid()

    print("Probabilities min:",
          probabilities.min().item())

    print("Probabilities max:",
              probabilities.max().item())

    print("Probabilities mean:",
              probabilities.mean().item())

    #Best class for every query
    scores,labels = probabilities.max(dim=-1)

    print("Max Scores min:",
              scores.min().item())

    print("Max Scores max:",
              scores.max().item())

    print("Max Scores mean:",
              scores.mean().item())

    detections=[]

    for batch_idx in range(images.shape[0]):
        image_scores = scores[batch_idx]
        image_labels = labels[batch_idx]
        image_boxes = pred_boxes[batch_idx]

        #Confidence filtering
        keep = image_scores >= confidence_threshold

        image_scores = image_scores[keep]
        image_labels = image_labels[keep]
        image_boxes = image_boxes[keep]

        #Convert normalized cxcywh to normalized xyxy
        image_boxes = box_cxcywh_to_xyxy(
          image_boxes
        )

        #Keep boxes within image boundaries
        image_boxes = image_boxes.clamp(
          min=0.0,
          max=1.0
        )

        detections.append({
          "boxes": image_boxes,
          "labels": image_labels,
          "scores": image_scores  
        })
    
    return detections

#Boxes: normalized xyxy [N,4]
def boxes_to_image_coordinated(
        boxes,
        image_height,
        image_width
):
    boxes = boxes.clone()

    boxes[:, [0,2]] *=image_width
    boxes[:,[1,3]] *=image_height

    return boxes

@torch.no_grad()
def inspect_top_predictions(model, images, targets, top_k=10):
    model.eval()

    outputs = model(images)

    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']

    probabilities = pred_logits.sigmoid()

    #Best classes for every query
    scores, labels = probabilities.max(dim=-1)

    for i in range(images.shape[0]):

      #Get top-k highest confidence queries
      top_scores,top_indices = torch.topk(
          scores[i],
          k=top_k
      )

      top_labels = labels[i][top_indices]
      top_boxes = pred_boxes[i][top_indices]

      #Convert cxcywh to xyxy
      top_boxes = box_cxcywh_to_xyxy(top_boxes)

      #keep boxes within the image
      top_boxes = top_boxes.clamp(
          min=0.0,
          max=1.0
      )

      print(f"\n Image {i}")
      print("GT Labels:", targets[i]["labels"].tolist())
      print("Top Scores:", top_scores.tolist())
      print("Top Labels:", top_labels.tolist())
      print("Top Boxes", top_boxes.tolist())

    return outputs

