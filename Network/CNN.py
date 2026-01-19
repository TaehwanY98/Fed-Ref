import torch
import torch.nn as nn
import torch.nn.functional as F

class FEMNIST_CNN(nn.Module):
    def __init__(self, num_classes=62):
        super(FEMNIST_CNN, self).__init__()
        
        # Conv layer 1: 1 → 32
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=5, 
            stride=1, 
            padding=0
        )
        
        # Conv layer 2: 32 → 64
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=0
        )

        # After two convs and pooling:
        # Input: 28×28
        # conv1 → 24×24 → maxpool → 12×12
        # conv2 → 8×8 → maxpool → 4×4
        # Flatten: 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))     # (batch, 32, 24, 24)
        out = F.max_pool2d(out, 2)        # (batch, 32, 12, 12)
        out = F.relu(self.conv2(out))     # (batch, 64, 8, 8)
        out = F.max_pool2d(out, 2)        # (batch, 64, 4, 4)

        out = out.reshape(x.size(0), -1)     # flatten → 1024
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class CINIC10_LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CINIC10_LightCNN, self).__init__()

        # 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     # 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 64 x 16 x 16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 128 x 8 x 8

        self.pool = nn.MaxPool2d(2, 2)  # downsampling by factor 2

        # After 3 poolings: 32 → 16 → 8 → 4
        # final feature map: 128 x 4 x 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 32×32×32
        x = self.pool(x)            # 32×16×16

        x = F.relu(self.conv2(x))   # 64×16×16
        x = self.pool(x)            # 64×8×8

        x = F.relu(self.conv3(x))   # 128×8×8
        x = self.pool(x)            # 128×4×4

        x = x.reshape(x.size(0), -1)  # flatten (2048)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example usage:
# model = CINIC10_LightCNN(num_classes=10)
# print(model)
