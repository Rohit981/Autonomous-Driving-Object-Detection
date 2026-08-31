import torch
import torch.nn as nn

# Common Convolution Backbone
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)

#DINO Backbone
class DINOBackbone(nn.Module):
    def __init__(self, 
                 in_channels=3):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBlock(in_channels,64),
            ConvBlock(64,128)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(128,256)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(256,512)
        )

        self.stage4 = nn.Sequential(
            ConvBlock(512,1024)
        )

    def forward(self,x):
        features = []

        x = self.stage1(x)

        #P3
        x = self.stage2(x)
        features.append(x)

        #P4
        x = self.stage3(x)
        features.append(x)

        #P5
        x = self.stage4(x)
        features.append(x)

        return features

class FeatureProjection(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dims=256):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            hidden_dims,
            kernel_size=1
        )
        
    def forward(self,x):
        return self.projection(x)