import argparse
import os
import numpy as np
import math
from d2l import torch as d2l
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
os.chdir('E:/GAN_DAMAGE')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
img_shape=(1,28,28)
cuda=True

class Discriminator(nn.Module):
    def __init__(self,channels_img,features_d):
        super(Discriminator,self).__init__()
        self.disc=nn.Sequential(
        nn.Conv2d(channels_img,features_d,kernel_size=(4,3),stride=(2,1),padding=1
                 ),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,(4,3),(2,1),1),
            self._block(features_d*2,features_d*4,(4,4),(2,2),1),
            self._block(features_d*4,features_d*8,(4,4),(2,2),1),
            self._block(features_d*8,features_d*4,(4,4),(2,2),1),
            self._block(features_d*4,features_d*2,(4,4),(2,2),1),
            nn.Conv2d(features_d*2,1,kernel_size=(4,2),stride=(2,2),padding=0),
            nn.Sigmoid(),
        )
    
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x):
        return self.disc(x)
    
z_dim,num_epochs,batch_size=1,1000,64
adversarial_loss=torch.nn.BCELoss()

discriminator=Discriminator(1,8)


discriminator.cuda()
adversarial_loss.cuda()
optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
x_data=np.load('d6_DATA.npy')
x_data=torch.tensor(x_data,dtype=torch.float32)
split=[x_data.shape[0],0]
dataset=torch.utils.data.TensorDataset(*(x_data,torch.ones(x_data.shape)))
train_dataset,test_dataset=torch.utils.data.random_split(dataset,split)
train_iter=d2l.load_array((train_dataset[:][0],train_dataset[:][1]),batch_size,is_train=True)

checkpoint=torch.load('discribe')
start_epoch=checkpoint['epoch']+1
discriminator.load_state_dict(checkpoint['model'])
optimizer_D.load_state_dict(checkpoint['optimizer'])
output=[]
for imgs,_ in train_iter:
    imgs=imgs.reshape(imgs.size(0),1,imgs.size(1),imgs.size(2))
    real_imgs=Variable(imgs.type(torch.Tensor)).to(d2l.try_gpu())
    out=discriminator(real_imgs)
    out=torch.mean(out)
    output.append(torch.detach(out.cpu()).numpy())

a=np.average(output)
print(a)