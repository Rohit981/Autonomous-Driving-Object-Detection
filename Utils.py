import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches

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

@torch.no_grad()
def inspect_predboxes_center(
   model,
   images,
   targets
):
      model.eval()
      outputs = model(images)
     
      pred_boxes = outputs["pred_boxes"]
      print("\n================ VALIDATION PREDICTED BOXES ================")
      print("\nPredicted Boxes:")
      print("min:", pred_boxes.min().item())
      print("max:", pred_boxes.max().item())
      print("mean:", pred_boxes.mean().item())
      print("std:", pred_boxes.std().item())
  
      print("\nPredicted widths/heights:")
      pred_wh = pred_boxes[...,2:4]
  
      print("pred_width mean:", pred_wh[...,0].mean().item())
      print("pred_height mean:", pred_wh[...,1].mean().item())
      print("pred_width max:", pred_wh[...,0].max().item())
      print("pred_height max:", pred_wh[...,1].max().item())
  
      #Validation GT box analysis
      print("\n================ VALIDATION GT BOXES =======================")
  
      for i in range(len(targets)):
        gt_boxes = targets[i]["boxes"]

        print(f"\nImages:{i}")

        print("GT boxes min:",
              gt_boxes.min().item())
        
        print("GT boxes max:",
                      gt_boxes.max().item())
        
        print("GT boxes mean:",
                      gt_boxes.mean().item())

        gt_wh = gt_boxes[...,2:4]

        print("GT Width mean:",
              gt_wh[...,0].mean().item())
        print("GT Height mean:",
                    gt_wh[...,1].mean().item())
        print("GT Width max:",
                    gt_wh[...,0].max().item())
        print("GT Height max:",
                    gt_wh[...,1].max().item())

      print("\n================ CENTER DISTRIBUTION =======================")

      pred_centers = pred_boxes[...,0:2]

      print("Pred Center X mean:",
            pred_centers[...,0].mean().item())
      print("Pred Center Y mean:",
                  pred_centers[...,1].mean().item())
      print("Pred Center X max:",
                  pred_centers[...,0].max().item())
      print("Pred Center Y max:",
                  pred_centers[...,1].max().item())
      print("Pred Center X min:",
            pred_centers[...,0].min().item())
      print("Pred Center Y min:",
            pred_centers[...,1].min().item())

      for i in range(len(targets)):
        gt_boxes = targets[i]["boxes"]
        gt_centers = gt_boxes[...,0:2]

        print(f"\nImages:{i}")

        print("GT center X min:",
              gt_centers[...,0].min().item())
        
        print("GT center X max:",
              gt_centers[...,0].max().item())
        
        print("GT Center Y min:",
              gt_centers[...,1].min().item())

        print("GT Center Y max:",
             gt_centers[...,1].max().item())
        print("GT Center X mean:",
              gt_centers[...,0].mean().item())
        print("GT Center Y mean:",
              gt_centers[...,1].mean().item())

      print("\n================ PREDICTED CENTER DISTRIBUTION ================")

      for i in range(pred_boxes.shape[0]):
          image_pred_boxes = pred_boxes[i]

          pred_centers = image_pred_boxes[:,0:2]

          print(f"\nImage {i}")

          print("Pred Center X mean:",
                pred_centers[:, 0].mean().item())

          print("Pred Center X min:",
                pred_centers[:, 0].min().item())

          print("Pred Center X max:",
                pred_centers[:, 0].max().item())

          print("Pred Center Y mean:",
                pred_centers[:, 1].mean().item())

          print("Pred Center Y min:",
                pred_centers[:, 1].min().item())

          print("Pred Center Y max:",
                pred_centers[:, 1].max().item())

          pred_wh = image_pred_boxes[:, 2:4]

          print("Pred Width mean:",
                pred_wh[:, 0].mean().item())

          print("Pred Height mean:",
                pred_wh[:, 1].mean().item())

          print("Pred Width max:",
                pred_wh[:, 0].max().item())

          print("Pred Height max:",
                pred_wh[:, 1].max().item())

@torch.no_grad()
def visualize_detection(
   model,
   images,
   targets,
   class_names,
   confidence_threshold=0.1,
   max_images=5
):

      model.eval()

      #Model Inference Mode
      outputs = model(images)

      pred_boxes = outputs['pred_boxes']
      pred_logits = outputs['pred_logits']

      probabilites = pred_logits.sigmoid()

      #Best class for every query
      scores,labels = probabilites.max(dim=-1)

      #Visualize each Image
      num_images = min(images.shape[0], max_images)

      for image_idx in range(num_images):
            image = images[image_idx].detach().cpu()

            #CHW-> HWC
            image = image.permute(1,2,0).numpy()

            mean = torch.tensor([0.485,0.456,0.406]).view(1,1,3)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,1,3)

            image = image * std.numpy() + mean.numpy()
            image = image.clip(0,1)

            image_height, image_width = image.shape[:2]

            #Create side by side figure
            fig, axes = plt.subplots(
            1,
            2,
            figsize=(16,7)
            )

            #LEFT: GROUND TRUTH
            axes[0].imshow(image)
            axes[0].set_title("Ground Truth")

            gt_boxes = targets[image_idx]['boxes'].detach().cpu()
            gt_labels = targets[image_idx]['labels'].detach().cpu()

            for box, label in zip(gt_boxes, gt_labels):

                  cx,cy,w,h = box.tolist()

                  #normalize cxcywh to pixel xywh
                  x = (cx - w/2) * image_width
                  y = (cy - h/2) * image_height

                  box_width = w * image_width
                  box_height = h * image_height

                  rectangle_patch = patches.Rectangle(
                  (x,y),
                  box_width,
                  box_height,
                  linewidth=2,
                  edgecolor='lime',
                  facecolor='none'
                  )

                  axes[0].add_patch(rectangle_patch)

                  class_id = int(label)

                  if class_id < len(class_names):
                        class_name = class_names[class_id]
                  else:
                        class_name = str(class_id)

                  axes[0].text(
                  x,
                  max(y-5,0),
                  class_name,
                  fontsize=10,
                  color="white",
                  backgroundcolor='green'
                  )

            axes[0].axis('off')

            #Right Prediction
            axes[1].imshow(image)
            axes[1].set_title(f"Predictions (threshold={confidence_threshold})")

            image_scores = scores[image_idx]
            image_labels = labels[image_idx]
            image_boxes = pred_boxes[image_idx]

            #Confidence filtering
            keep = image_scores >= confidence_threshold

            image_scores = image_scores[keep]
            image_labels = image_labels[keep]
            image_boxes = image_boxes[keep]

            for box_pred,label_pred,score in zip(
            image_boxes,
            image_labels,
            image_scores
            ):
                  cx,cy,w,h = box_pred.tolist()

                  x = (cx - w/2) * image_width
                  y = (cy - h/2) * image_height

                  box_width = w * image_width
                  box_height = h * image_height

                  rectangle_patch = patches.Rectangle(
                        (x,y),
                        box_width,
                        box_height,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                  )

                  axes[1].add_patch(rectangle_patch)

                  class_id = int(label_pred)

                  if class_id < len(class_names):
                        class_name = class_names[class_id]
                  else:
                        class_name = str(class_id)

                  label_text = f"{class_name} {score.item():2f}"

                  axes[1].text(
                        x,
                        max(y-5,0),
                        label_text,
                        fontsize=10,
                        color='white',
                        backgroundcolor='red'
                  )

            axes[1].axis('off')

            plt.tight_layout()
            plt.imshow()
            




      

