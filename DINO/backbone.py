import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import(
    vit_b_16,
    ViT_B_16_Weights
)

#DINO Backbone
class DINOBackbone(nn.Module):
    def __init__(self, 
                 pretrained=True,
                 hidden_dim=256):
        super().__init__()

        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT

            backbone = vit_b_16(
                weights=weights
            )

        else:
            backbone = vit_b_16(
                weights=None
            )

        #VIT Components
        self.hidden_dim = hidden_dim
        self.conv_proj = backbone.conv_proj
        self.class_token = backbone.class_token

        self.encoder_layer = backbone.encoder.layers
        self.encoder_dropout = backbone.encoder.dropout
        self.encoder_ln = backbone.encoder.ln

        #Pretrained positional embedding
        self.pos_embedding = backbone.encoder.pos_embedding

        #FPN
        self.p4_projection = nn.Conv2d(
            768,
            hidden_dim,
            kernel_size=1
        )

        self.p3_upsample = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        )

        self.p5_downsample = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def interpolate_positional_embedding(
            self,
            x,
            height,
            width
    ):
        #Original positional embedding
        cls_pos = self.pos_embedding[:,:1,:]
        patch_pos = self.pos_embedding[:,1:,:]

        #[1,196,768]
        patch_pos = patch_pos.reshape(
            1,
            14,
            14,
            768
        )

        #[1,768,14,14]
        patch_pos = patch_pos.permute(
            0,
            3,
            1,
            2
        )

        #Resize 14x14 to current feature resolution
        patch_pos = F.interpolate(
            patch_pos,
            size=(height,width),
            mode="bicubic",
            align_corners=False
        )

        #[1,768,H,W]
        patch_pos = patch_pos.permute(
            0,
            2,
            3,
            1
        )

        #[1,H*W,768]
        patch_pos = patch_pos.reshape(
            1,
            height*width,
            768
        )

        #Add CLS positional embedding back
        pos_embedding = torch.cat(
            [
                cls_pos,
                patch_pos
            ],
            dim=1
        )

        return pos_embedding

    def forward(self,x):

        #Patch Embedding
        x = self.conv_proj(x)

        #[B,768,40,40]
        B,C,H,W = x.shape

        #Flatten Patches
        x = x.flatten(2)

        #[B,768,1600]
        x = x.permute(
            0,2,1
        )

        # [B,1600,768]
        #Add class Token

        class_token = self.class_token.expand(
            B,
            -1,
            -1
        )

        x = torch.cat(
            [class_token,x],
            dim=1
        )

        #[B,1601,768]
        #Interpolate Positional Embeddings
        pos_embedding = self.interpolate_positional_embedding(x,H,W)

        #[1,1601,768]
        x = x + pos_embedding

        x = self.encoder_dropout(
            x
        )

        #Transformer
        x = self.encoder_layer(
            x
        )
        
        x = self.encoder_ln(
            x
        )
       
        #Remove CLS Token
        x= x[:,1:,:]

        #Convert token to feature map
        x = x.permute(
            0,2,1
        )

        x = x.reshape(
            B,
            C,
            H,
            W
        )

        #[B,768,40,40]
        #P4
        p4 = self.p4_projection(x)

        #[B,256,40,40]
        #P3
        p3 = self.p3_upsample(p4)

        #[B,256,80,80]
        #P5
        p5 = self.p5_downsample(p4)

        return[
            p3,
            p4,
            p5
        ]

# if __name__ == "__main__":

#     model = DINOBackbone(
#         pretrained=True,
#         hidden_dim=256
#     )

#     images = torch.randn(
#         2,
#         3,
#         640,
#         640
#     )

#     features = model(images)

#     for i, feature in enumerate(features):

#         print(
#             f"P{i+3}:",
#             feature.shape
#         )


# class FeatureProjection(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  hidden_dims=256):
#         super().__init__()

#         self.projection = nn.Conv2d(
#             in_channels,
#             hidden_dims,
#             kernel_size=1
#         )
        
#     def forward(self,x):
#         return self.projection(x)