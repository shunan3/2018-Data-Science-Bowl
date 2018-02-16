import torch
import torch.nn as nn

class vanilla(nn.Module):
    def __init__(self):
        super(vanilla, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.upsample(input=x, size=input_spatial_dim, 'bilinear')
        return x
