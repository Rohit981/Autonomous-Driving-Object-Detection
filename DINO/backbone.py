import torch
import torch.nn as nn

from torchvision.models import(
    resnet50,
    ResNet50_Weights
)

#DINO Backbone
class DINOBackbone(nn.Module):
    def __init__(self, 
                 pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT

            backbone = resnet50(
                weights=weights
            )

        else:
            backbone = resnet50(
                weights=None
            )

        #Stem
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        #ResNet stages
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self,x):

        #Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #C2
        x = self.layer1(x)

        #C3
        x = self.layer2(x)
        feature_p3 = x

        #C4
        x = self.layer3(x)
        feature_p4 = x

        #C5
        x = self.layer4(x)
        feature_p5 = x

        return [
            feature_p3,
            feature_p4,
            feature_p5
        ]

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