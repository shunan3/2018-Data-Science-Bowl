import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch as t
from torch.utils import data
from torchvision import transforms as tsf

TRAIN_PATH = './train.pth'
TEST_PATH = './test.tph'

import os
from pathlib import Path
from PIL import Image
from skimage import io
import numpy as np
from tqdm import tqdm
import torch as t


def process(file_path, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in files:
        item = {}
        imgs = []
        for image in (file/'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs)==1
        if img.shape[2]>3:
            assert(img[:,:,3]!=255).sum()==0
        img = img[:,:,:3]

        if has_mask:
            mask_files = list((file/'masks').iterdir())
            masks = None
            for ii,mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask!=0)]==255).all()
                if masks is None:
                    H,W = mask.shape
                    masks = np.zeros((len(mask_files),H,W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask!=0] == 255).all()
            for ii,mask in enumerate(masks):
                masks[ii] = mask/255 * (ii+1)
            mask = masks.sum(0)
            item['mask'] = t.from_numpy(mask)
        item['name'] = str(file).split('/')[-1]
        item['img'] = t.from_numpy(img)
        datas.append(item)
    return datas

# You can skip this if you have alreadly done it.
test = process('./data/stage1_test/',False)
t.save(test, TEST_PATH)
train_data = process('./data/stage1_train/')



import PIL
class Dataset():
    def __init__(self,data,source_transform,target_transform):
        self.datas = data
#         self.datas = train_data
        self.s_transform = source_transform
        self.t_transform = target_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
        img = self.s_transform(img)
        mask = self.t_transform(mask)
        return img, mask
    def __len__(self):
        return len(self.datas)
s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((224,224)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
]
)
t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((224,224),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)
dataset = Dataset(train_data,s_trans,t_trans)
dataloader = t.utils.data.DataLoader(dataset,num_workers=2,batch_size=4)


def soft_dice_loss(inputs, targets):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score



import torch.nn as nn
import torchvision.models as models


class FCN_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(FCN_32s, self).__init__()
        
        # Load the model with convolutionalized
        # fully connected layers
        vgg16 = models.vgg16(pretrained=True)
        
        # Copy all the feature layers as is
        self.features = vgg16.features
        
        # TODO: check if Dropout works correctly for
        # fully convolutional mode
        
        # Remove the last classification 1x1 convolution
        # because it comes from imagenet 1000 class classification.
        # We will perform classification on different classes
        fully_conv = list(vgg16.classifier.children())
        fully_conv = fully_conv[:-1]
        self.fully_conv = nn.Sequential(*fully_conv)
        
        # Get a new 1x1 convolution and randomly initialize
        score_32s = nn.Linear(4096, num_classes)
        self._normal_initialization(score_32s)
        self.score_32s = score_32s
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fully_conv(x)
        x = self.score_32s(x)
        
        #x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x


model = FCN_32s(num_classes=1).cuda()
optimizer = t.optim.Adam(model.parameters(),lr = 1e-3)

for epoch in range(2):
    for x_train, y_train  in dataloader:
        x_train = t.autograd.Variable(x_train).cuda())
        y_train = t.autograd.Variable(y_train).cuda())
        optimizer.zero_grad()
        o = model(x_train)
        loss = soft_dice_loss(o, y_train)
        loss.backward()
        optimizer.step()



class TestDataset():
    def __init__(self,path,source_transform):
        self.datas = t.load(path)
        self.s_transform = source_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        img = self.s_transform(img)
        return img
    def __len__(self):
        return len(self.datas)

testset = TestDataset(TEST_PATH, s_trans)
testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=2)


model = model.eval()
for data in testdataloader:
    data = t.autograd.Variable(data, volatile=True).cuda())
    o = model(data)
    break

tm=o[1][0].data.cpu().numpy()

test_img = data[1].data.cpu().permute(1,2,0).numpy()*0.5+0.5
test_mask = tm



numpy.save(test_img, test_img)
numpy.save(test_mask, test_mask)


t.save(model, fcn_32s_model)
