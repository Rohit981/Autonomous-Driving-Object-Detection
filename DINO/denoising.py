import torch
import torch.nn as nn
from attention_mask import build_attention_mask

class ContrastiveDenoising(nn.Module):
    def __init__(self,
                 num_classes,
                 hidden_dim=256,
                 num_dn_groups=5,
                 num_queries=300,
                 label_noise_ratio=0.5,
                 box_noise_scale=0.4):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.num_dn_groups = num_dn_groups
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.num_dn_groups = num_dn_groups
        self.num_queries = num_queries

        #Label embedding used to create
        #denoising content queries
        self.label_embedding = nn.Embedding(
            num_classes,
            hidden_dim
        )

    #Repeate positive and negative DN copies
    def repeat_dn_groups(
            self,
            labels,
            boxes
    ):
        num_objects = labels.shape[0]

        #Positive + negative copies
        copies_per_group = 2

        total_copies = (
            self.num_dn_groups
            * copies_per_group
        )

        repeated_labels = labels.repeat(
            total_copies
        )

        repeated_boxes = boxes.repeat(
            total_copies,
            1
        )

        #Track which original GT object
        #Each DN query belongs to
        target_indices = torch.arange(
            num_objects,
            device=labels.device
        ).repeat(
            total_copies
        )

        return(
            repeated_labels,
            repeated_boxes,
            target_indices
        )

    #Create Positive and negative box noise
    def add_contrastive_box_noise(
            self,
            boxes
    ):
        num_boxes = boxes.shape[0]

        half = num_boxes//2

        noisy_boxes = boxes.clone()

        #Positive queries
        if half > 0:
            positive_boxes = boxes[
                :half
            ]

            centers = positive_boxes[
                :,
                :2
            ]

            sizes = positive_boxes[
                :,
                2:
            ]

            noise = (torch.rand_like(centers) * 2 - 1)

            noisy_centers = (
                centers
                +
                noise
                *sizes
                *self.box_noise_scale
                *0.5
            )

            noisy_boxes[
                :half,
                :2
            ] = noisy_centers

        #Negative Queries
        if half < num_boxes:

            negative_boxes = boxes[
                half:
            ]

            centers = negative_boxes[
                :,
                :2
            ]

            sizes = negative_boxes[
                :,
                2:
            ]

            noise = (
                torch.rand_like(centers) * 2 - 1
            )

            noisy_centers = (
                centers
                +
                noise
                *sizes
                *self.box_noise_scale
                * 2.0
            )

            noisy_boxes[
                half:,
                :2
            ] = noisy_centers

        return noisy_boxes.clamp(
            min=0.0,
            max=1.0
        )


    #Noisy Labels
    def add_label_noise(
            self,
            labels
    ):
        if self.label_noise_ratio == 0:
            return labels

        noisy_labels = labels.clone()

        #Randomly choose labels to corrupt
        noise_mask = (
            torch.rand(
                labels.shape,
                device=labels.device
            )
            < self.label_noise_ratio
        )

        #Generate random replacement labels
        random_labels = torch.randint(
            low=0,
            high=self.num_classes,
            size=labels.shape,
            device=labels.device
        )

        noisy_labels[
            noise_mask
        ] = random_labels[
            noise_mask
        ]

        return noisy_labels

    #Box Noise
    def add_box_noise(
            self,
            boxes
    ):

        noisy_boxes = boxes.clone()

        if self.box_noise_scale == 0:
            return noisy_boxes

        #Seperate centre and size
        centers = boxes[:,:2]
        sizes = boxes[:,2:]

        #Generate random noise
        center_noise = (
            torch.rand_like(centers) * 2 - 1
        )

        size_noise = (
            torch.rand_like(sizes) * 2 - 1
        )

        #Scale center noise relative to box size
        centers = (
            centers
            +
            center_noise
            * sizes
            * self.box_noise_scale
        )

        #Pertub width and height
        sizes = (
            sizes
            *
            (
                1
                +
                size_noise
                *self.box_noise_scale
            )
        )

        noisy_boxes = torch.cat(
            [
                centers,
                sizes
            ],
            dim=-1
        )

        #Keep normalized coordinates valid
        noisy_boxes = noisy_boxes.clamp(
            min=0.0,
            max=1.0
        )

        return noisy_boxes
    
    #Generate DN queries for one Image
    def prepare_single_image(
            self,
            labels,
            boxes
    ):
        if labels.numel() == 0:
            empty_queries = torch.zeros(
                0,
                self.hidden_dim,
                device=boxes.device
            )

            empty_boxes = torch.zeros(
                0,
                4,
                device=boxes.device
            )

            empty_indices = torch.zeros(
                0,
                dtype=torch.long,
                device=boxes.device
            )

            return (
                empty_queries,
                empty_boxes,
                empty_indices
            )

        #Repeat GT objects for DN groups
        (
            repeated_labels,
            repeated_boxes,
            target_indices
        ) = self.repeat_dn_groups(
            labels,
            boxes
        )

        #Add label noise
        noisy_labels = self.add_label_noise(
            repeated_labels
        )

        #Add contrastive box noise
        noisy_boxes = self.add_box_noise(
            repeated_boxes  
        )

        #Labels become DN content queries
        dn_queries = self.label_embedding(
            noisy_labels
        )

        return(
            dn_queries,
            noisy_boxes,
            target_indices
        )

    #Batch Padding add to the images
    def pad_dn_queries(
            self,
            dn_queries_list,
            dn_boxes_list
    ):
        batch_size = len(
            dn_queries_list
        )

        max_dn_queries = max(
            query.shape[0]
            for query in dn_queries_list
        )

        device = dn_queries_list[0].device

        padded_queries = torch.zeros(
            batch_size,
            max_dn_queries,
            self.hidden_dim,
            device=device
        )

        padded_boxes = torch.zeros(
            batch_size,
            max_dn_queries,
            4,
            device=device
        )

        for batch_idx in range(batch_size):

            num_queries = (
                dn_queries_list[
                    batch_idx
                ].shape[0]
            )

            if num_queries == 0:
                continue

            padded_queries[
                batch_idx,
                :num_queries
            ] = dn_queries_list[
                batch_idx
            ]

            padded_boxes[
                batch_idx,
                :num_queries
            ] = dn_boxes_list[
                batch_idx
            ]

        return (
            padded_queries,
            padded_boxes
        )

    #Helper that processes all images
    def prepare_dn_queries(
            self,
            targets,
            device
    ):
        batch_size = len(targets)

        dn_queries = []
        dn_boxes = []
        num_objects = []

        for target in targets:

            labels = target[
                "labels"
            ].to(device)

            boxes = target[
                "boxes"
            ].to(device)

            #Store number of objects
            num_objects.append(
                labels.shape[0]
            )

            #Add Noise
            noisy_labels = self.add_label_noise(
                labels
            )

            noisy_boxes = self.add_box_noise(
                boxes
            )

            #Convert labels -> query embedding
            query_embeddings = self.label_embedding(
                noisy_labels
            )

            dn_queries.append(
                query_embeddings
            )

            dn_boxes.append(
                noisy_boxes
            )

        return (
            dn_queries,
            dn_boxes,
            num_objects
        )

    def forward(
            self,
            targets,
            device
    ):

        dn_queries_list = []
        dn_boxes_list = []
        target_indices_list = []

        for target in targets:

            labels = target[
                "labels"
            ].to(device)

            boxes = target[
                "boxes"
            ].to(device)

            (
                dn_queries,
                noisy_boxes,
                target_indices
            ) = self.prepare_single_image(
                labels,
                boxes
            )

            dn_queries_list.append(
                dn_queries
            )

            dn_boxes_list.append(
                noisy_boxes
            )

            target_indices_list.append(
                target_indices
            )

        (
            dn_queries,
            dn_boxes
        ) = self.pad_dn_queries(
            dn_queries_list,
            dn_boxes_list
        )

        num_dn_queries = dn_queries.shape[1]

        #Build decoder self-attention mask
        attn_mask = build_attention_mask(
            num_dn_queries=num_dn_queries,
            device=device,
            num_queries=self.num_queries
        )

        dn_meta = {
            "num_dn_groups":
                self.num_dn_groups,

            "num_dn_queries":
                dn_queries.shape[1],

            "target_indices":
                target_indices_list
        }
            

        return {
            "dn_queries": dn_queries,
            "dn_boxes": dn_boxes,
            "attn_mask": attn_mask,
            "dn_meta": dn_meta
        }

#Test
# NUM_CLASSES = 10
# DEVICE = "cuda"
# targets = [
#     {
#         "labels": torch.tensor(
#             [1,3,5]
#         ),
#         "boxes": torch.rand(
#             3,
#             4
#         )
#     },
#     {
#         "labels": torch.tensor(
#             [2,4]
#         ),

#         "boxes": torch.rand(
#             2,
#             4
#         )
#     }
# ]

# denoising = ContrastiveDenoising(
#     NUM_CLASSES
# ).to(DEVICE)

# denoise = denoising(
#     targets,
#     DEVICE
# )

# dn_queries = denoise["dn_queries"]
# dn_boxes = denoise["dn_boxes"]
# dn_meta = denoise["dn_meta"]

# print("dn_queries:", dn_queries)
# print("dn_boxes:", dn_boxes)
# print("dn_meta:", dn_meta)

