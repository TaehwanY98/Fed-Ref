from torchvision.models.resnet import resnet50
import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet = resnet50(False)
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 7), nn.Sigmoid())
        for m in [m for m in self.children()][:-2]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        return self.resnet(x)