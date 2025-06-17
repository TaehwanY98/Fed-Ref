from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch.nn as nn
class MobileNet(nn.Module):
    def __init__(self, outdim=7,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = mobilenet_v3_small()
        self.net.classifier = nn.Sequential(nn.Linear(576, outdim), nn.Sigmoid())
        for m in [m for m in self.children()][:-2]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        return self.net(x)