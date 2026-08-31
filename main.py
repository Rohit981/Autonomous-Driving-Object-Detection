from ultralytics import YOLO
import os
import Config
import Dataset as dataset
import transform

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

    #Test data loader
    train_dataset = dataset.BDD100kDataset(
        image_dir=config.BDD100k_img_dir,
        labels_dir=config.BDD100k_label_dir,
        classes_names=config.CLASS_NAMES,
        transform=train_transform
    )


if __name__ == "__main__":
    main()
