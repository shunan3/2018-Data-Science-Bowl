import torch
import torch.nn as nn

class vgg16(nn.Module):
    def __init__(self):
        super(FCN_VGG16, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d (3, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (64, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                                     nn.Conv2d (64, 128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (128, 128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                                     nn.Conv2d (128, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (256, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (256, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                                     nn.Conv2d (256, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
                                     nn.Conv2d (512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d (512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
                                    )
    
        self.fully_conv = nn.Sequential(nn.Conv2d (512, 4096, kernel_size=7, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d (4096, 4096, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.5),
                                    nn.Conv2d (4096, 1, kernel_size=1, padding=0)                                   
                                    )
    
    
    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.feature(x)
        x = self.fully_conv(x)
       
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        return x
