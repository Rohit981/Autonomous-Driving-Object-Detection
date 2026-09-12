from ultralytics import YOLO
import os
import Config
import Dataset as dataset
from torch.utils.data import DataLoader,Subset
import transform
from DINO import loss,model
from DINO.matcher import HungarianMatcher
from Trainer import ModelTrainer
from Utils import (Visualize_loss_acc, 
                   get_detection, 
                   inspect_top_predictions,
                   inspect_predboxes_center,
                   visualize_detection)
import torch

def main():
    # model = YOLO('yolo11n.pt')

    # #Check if directory exist
    # dir_path = "Data/BDD100k/data.yaml"

    # if os.path.exists(dir_path):
    #     print("Path exists")
    # else:
    #     print("Path Not exists")


    # #Train the model on Bdk100k image set
    # results = model.train(data = dir_path, epochs=100, imgsz=640)

    #Initialize config and train transform
    config = Config.DATA_CONFIG()
    train_transform = transform.get_train_transforms()
    val_transform = transform.get_val_transforms()

    #Train dataset
    train_dataset = dataset.BDD100kDataset(
        image_dir=config.BDD100k_train_img_dir,
        labels_dir=config.BDD100k_train_label_dir,
        classes_names=config.CLASS_NAMES,
        transform=train_transform
    )

    #Validation Dataset
    val_dataset = dataset.BDD100kDataset(
        image_dir=config.BDD100k_val_img_dir,
        labels_dir=config.BDD100k_val_label_dir,
        classes_names=config.CLASS_NAMES,
        transform=val_transform
    )

    train_subset = Subset(
        train_dataset,
        range(50)
    )

    val_subset = Subset(
        val_dataset,
        range(25)
    )

    #Train DataLoader
    train_dataloader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )


    #Intialize Loss
    hungarian_matcher = HungarianMatcher()
    criterion = loss.DINOLoss(
        num_classes=len(config.CLASS_NAMES),
        matcher=hungarian_matcher
    )

    Dino = model.DINO(
        num_classes=len(config.CLASS_NAMES)
    )

    trainer = ModelTrainer(
        model=Dino,
        criterion=criterion,
        device=config.device,
        learning_rate=config.learning_rate
    )

    trainer.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=config.n_epochs
    )


    #Run Detection after training
    images, targets = next(iter(val_dataloader))

    images = images.to(config.device)
    targets = trainer.move_targets_to_device(targets)

    inspect_predboxes_center(
        model=trainer.model,
        images=images,
        targets=targets
    )

    val_detections = get_detection(
        model=trainer.model,
        images=images,
        confidence_threshold=0.1
    )

    visualize_detection(
        model=trainer.model,
        images=images,
        targets=targets,
        class_names=config.CLASS_NAMES,
        confidence_threshold=0.1,
        max_images=5
    )

    for i, detection in enumerate(val_detections):
        print(f"\nimage:{i}")

        print("Validation GT:", targets[i]["labels"].tolist())
        print("Validation Pred labels:", detection["labels"].tolist())
        print("Validation Scores:", detection["scores"].tolist())

    #Inspect strongest 10 queries
    inspect_top_predictions(
        trainer.model,
        images,
        targets
    )


     #Run Detection after training
    train_images, train_targets = next(iter(train_dataloader))

    train_images = train_images.to(config.device)
    train_targets = trainer.move_targets_to_device(train_targets)

    inspect_predboxes_center(
        model=trainer.model,
        images=train_images,
        targets=train_targets
    )

    train_detections = get_detection(
        model=trainer.model,
        images=train_images,
        confidence_threshold=0.1
    )

    for i, detection in enumerate(train_detections):
        print(f"\nimage:{i}")

        print("Trainer GT:", train_targets[i]["labels"].tolist())
        print("Trainer Pred labels:", detection["labels"].tolist())
        print("Trainer Scores:", detection["scores"].tolist())

    #Inspect strongest 10 queries
    inspect_top_predictions(
        trainer.model,
        train_images,
        train_targets
    )
    


    Visualize_loss_acc(train_loss=trainer.train_loss,
                       val_loss=trainer.val_loss,
                       train_class_loss=trainer.train_class_loss,
                       val_class_loss=trainer.val_class_loss,
                       train_bbox_loss=trainer.train_bbox_loss,
                       val_bbox_loss=trainer.val_bbox_loss,
                       train_giou_loss=trainer.train_giou_loss,
                       val_giou_loss=trainer.val_giou_loss)


if __name__ == "__main__":
    main()
