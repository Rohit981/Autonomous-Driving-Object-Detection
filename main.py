from ultralytics import YOLO
import os
import Config
import Dataset as dataset
from torch.utils.data import DataLoader,Subset
import transform
from DINO import loss,model
from DINO.matcher import HungarianMatcher
from Trainer import ModelTrainer
from Utils import Visualize_loss_acc

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

    # images, targets = next(iter(train_dataloader))

    # images = images.to(config.device)

    # targets = trainer.move_targets_to_device(targets)

    # trainer.model.train()

    # trainer.optimizer.zero_grad(
    #     set_to_none=None
    # )
   
    # outputs = trainer.model(
    #     images,
    #     targets
    # )

    # print("Forward successful")

    # losses = trainer.criterion(
    #     class_logits=outputs["pred_logits"],
    #     pred_boxes=outputs["pred_boxes"],
    #     targets=targets,

    #     auxiliary_class_logits=outputs["aux_class_logits"],
    #     auxiliary_boxes=outputs["aux_boxes"],

    #     dn_class_logits=outputs["dn_class_logits"],
    #     dn_boxes=outputs["dn_boxes"],
    #     dn_meta=outputs["dn_meta"]
    # )

    # loss_value = losses['loss_total']
    # print("Loss:", loss_value.item())
    # loss_value.backward()
    # print("BackPropogation successful")
    # trainer.optimizer.step()
    # print("Optimizer step successful")

    trainer.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=config.n_epochs
    )

    Visualize_loss_acc(train_loss=trainer.train_loss,
                       val_loss=trainer.val_loss)


if __name__ == "__main__":
    main()
