import torch

class DATA_CONFIG:
    #CLASS NAMES-----------------------------------------
    CLASS_NAMES = {
        "person": 0,
        "rider": 1,
        "car": 2,
        "bus": 3,
        "truck": 4,
        "bike": 5,
        "motor": 6,
        "traffic light": 7,
        "traffic sign": 8,
        "train": 9,
    }

    #Image and Label Dir for BDD100k
    BDD100k_train_img_dir = "Data/BDD100k/train/images"
    BDD100k_train_label_dir = "Data/BDD100k/train/labels"

    BDD100k_val_img_dir = "Data/BDD100k/val/images"
    BDD100k_val_label_dir = "Data/BDD100k/val/labels"

    device : torch.cuda = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-4
    n_epochs : int = 100
    batch_size: int = 1
    num_workers: int = 0