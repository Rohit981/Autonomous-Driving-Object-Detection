import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange


class ModelTrainer(nn.Module):
    def __init__(self,
                 model,
                 criterion,
                 device,
                 scheduler=None,
                 learning_rate=0.0):
        super().__init__()

        self.model = model.to(device)

        self.criterion = criterion

        self.scheduler = scheduler

        self.device = device

        self.learning_rate = learning_rate

        self.start_epoch = 0

        self.set_optimizer()
        self.set_lr_scheduler()

    def move_targets_to_device(
            self,
            targets
    ):
        targets = [
            {
                key: value.to(self.device)
                for key, value in target.items()
            }

            for target in targets   
        ]

        return targets

    def set_optimizer(self):
        self.optimizer = optim.AdamW(
           [
               {
                   "params": self.model.backbone.parameters(),
                   "lr": 1e-5
               },
               {
                   "params": [
                       p
                       for name,p in self.model.named_parameters()
                       if not name.startswith("backbone.")
                   ],
                   "lr": 1e-4
               }
           ],
            weight_decay=1e-4
        )

    def set_lr_scheduler(self):
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=40,
            gamma=0.1
        )

    def train_epoch(
            self,
            train_loader
    ):
        self.model.train()

        total_loss = 0.0

        total_class_loss = 0.0

        total_bbox_loss = 0.0

        total_giou_loss = 0.0

        num_batches = len(train_loader)

        for batch_idx, (
            images,
            targets
        ) in enumerate(tqdm(train_loader, leave=False, desc="Training")):

            #Move images to device
            images = images.to(self.device)

            #Move detection targets
            targets = self.move_targets_to_device(targets)

            #Clear gradient
            self.optimizer.zero_grad()

            #Forward Pass
            outputs = self.model(
                images,
                targets
            )

            losses = self.criterion(
                class_logits = 
                    outputs["pred_logits"],

                pred_boxes = 
                    outputs["pred_boxes"],

                targets = 
                    targets,

                auxiliary_class_logits=
                    outputs["aux_class_logits"],

                auxiliary_boxes = 
                    outputs["aux_boxes"],

                dn_class_logits = 
                    outputs["dn_class_logits"],

                dn_boxes = 
                    outputs["dn_boxes"],

                dn_meta = 
                    outputs["dn_meta"]
            )

            loss = losses[
                "loss_total"
            ]

            #Backpropogation
            loss.backward()

            #Optimal gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=0.1
            )

            #Update parameters
            self.optimizer.step()

            #Store losses
            total_loss += loss.item()

            total_class_loss += (
                losses["loss_class"].item()
            )

            total_bbox_loss += (
                losses["loss_bbox"].item()
            ) 

            total_giou_loss += (
                losses["loss_giou"].item()
            )

            if (
                batch_idx % 10 == 0
            ):
                print(
                    f"Batch"
                    f"[{batch_idx}/{num_batches}] |"
                    f"Loss: {loss.item(): .4f}"
                )

        return {
            "loss":
                total_loss/num_batches,

            "loss_class":
                total_class_loss/num_batches,

            "loss_bbox":
                total_bbox_loss/num_batches,

            "loss_giou":
                total_giou_loss/num_batches
        }

    #Validation
    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader
    ):
        self.model.eval()

        total_loss = 0.0

        for images,targets in tqdm(val_loader, leave=False, desc="Validation"):
            images = images.to(self.device)

            targets = self.move_targets_to_device(targets)

            #No DN during validation
            outputs = self.model(
                images
            )

            losses = self.criterion(
                class_logits = 
                    outputs['pred_logits'],

                pred_boxes=
                    outputs['pred_boxes'],

                targets=
                    targets
            )

            total_loss += (
                losses['loss_total'].item()
            )

        return(
            total_loss/ len(val_loader)
        )

    #Full training loop
    def fit(
            self,
            train_loader,
            val_loader,
            num_epochs
    ):
        history = []
        pbar = trange(self.start_epoch,num_epochs,leave=False,desc="Epoch")

        for epoch in pbar:

            print(
                f"\nEpoch"
                f"{epoch + 1}/{num_epochs}"
            )

            train_metrics = self.train_epoch(
                train_loader
            )

            val_loss = self.validate_epoch(
                val_loader
            )

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Train Loss: "
                f"{train_metrics['loss']:.4f}"
            )

            print(
                f"Val loss: "
                f"{val_loss:.4f}"
            )

            history.append(
                {
                    "epoch":
                        epoch + 1,

                    "train_loss":
                        train_metrics["loss"],

                    "val_loss":
                        val_loss
                }
            )

        return history