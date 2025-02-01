from torchvision.models.resnet import resnet50
import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet = resnet50(False)
        self.fcl = nn.Sequential(nn.Linear(1000, 7), nn.Softmax(dim=1))
    def forward(self, x):
        return self.fcl(self.resnet(x))