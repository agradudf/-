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
        
class Generator(nn.Module):
    def __init__(self,z_dim,channels_img,features_g):
        super(Generator,self).__init__()
        self.g=nn.Sequential(
            self._block_first(z_dim,features_g*16,(4,4),(1,1)),
            self._block(features_g*16,features_g*32,(4,4),(2,2),1),
            self._block(features_g*32,features_g*16,(4,4),(2,2),1),
            self._block(features_g*16,features_g*8,(4,2),(2,1),1),
            self._block(features_g*8,features_g*4,(4,4),(2,2),1),
            self._block(features_g*4,features_g*2,(4,4),(2,2),1),
            nn.ConvTranspose2d(
                features_g*2,channels_img,kernel_size=(4,3),stride=(2,1),padding=1,
            ),
            nn.Tanh(),
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),      
        )
    def _block_first(self,in_channels,out_channels,kernel_size,stride):
         return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),      
        )
    def forward(self,x):
        #print(self.g(x).shape)
        return self.g(x)

z_dim,num_epochs,batch_size=1,200,64
adversarial_loss=torch.nn.BCELoss()
generator=Generator(z_dim,1,8)
discriminator=Discriminator(1,8)

generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()

x_data=np.load('b1_DATA.npy')
x_data=torch.tensor(x_data,dtype=torch.float32)
print(x_data.shape)
split=[800,259]
dataset=torch.utils.data.TensorDataset(*(x_data,torch.ones(x_data.shape)))
train_dataset,test_dataset=torch.utils.data.random_split(dataset,split)
torch.save(test_dataset[:][0],'test_x')
train_iter=d2l.load_array((train_dataset[:][0],train_dataset[:][1]),batch_size,is_train=True)
test_iter=d2l.load_array((test_dataset[:][0],test_dataset[:][1]),batch_size,is_train=True)

optimizer_G=torch.optim.Adam(generator.parameters(),lr=0.0001,betas=(0.5,0.999))
optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.0001,betas=(0.5,0.999))
animator=d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[1,num_epochs],ylim=[1e-3,1e1],legend=['g_loss','d_loss','t_loss'])
tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor



for epoch in range(num_epochs):
    gr_loss=[]
    dr_loss=[]
    tr_loss=[]
    rl_loss=[]
    for imgs,_ in train_iter:
        imgs=imgs.reshape(imgs.size(0),1,imgs.size(1),imgs.size(2))
        valid=Variable(torch.Tensor(imgs.size(0),1,1,1).fill_(1.0),requires_grad=False).to(d2l.try_gpu())
        fake=Variable(torch.Tensor(imgs.size(0),1,1,1).fill_(0.0),requires_grad=False).to(d2l.try_gpu())
        
        real_imgs=Variable(imgs.type(torch.Tensor)).to(d2l.try_gpu())
        optimizer_G.zero_grad()
        z=torch.randn((imgs.shape[0],z_dim,1,1)).to(d2l.try_gpu())
        gen_imgs=generator(z)
        g_loss=adversarial_loss(discriminator(gen_imgs),valid)
        gr_loss.append(g_loss.item())
        g_loss.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        real_loss=adversarial_loss(discriminator(real_imgs),valid)
        fake_loss=adversarial_loss(discriminator(gen_imgs.detach()),fake)
        d_loss=(real_loss+fake_loss)/2
        dr_loss.append(d_loss.item())
        rl_loss.append(real_loss.item())
        d_loss.backward()
        optimizer_D.step()
        
    g_train_loss=np.average(gr_loss)
    d_train_loss=np.average(dr_loss)
    re_loss=np.average(rl_loss)
    loss_temp=1
    for x,_ in test_iter:
        discriminator.eval()
        with torch.no_grad():
            x=x.reshape(x.size(0),1,x.size(1),x.size(2))
            v=Variable(torch.Tensor(x.size(0),1,1,1).fill_(1.0),requires_grad=False).to(d2l.try_gpu())
            r_x=Variable(x.type(torch.Tensor)).to(d2l.try_gpu())
            r_loss=adversarial_loss(discriminator(r_x),v)
            tr_loss.append(r_loss.item())
    discriminator.train()
    test_loss=np.average(tr_loss)
    if epoch==0 or (epoch+1)%2==0:
        animator.add(epoch+1,(g_train_loss,d_train_loss,test_loss))
        if epoch>40:
            if loss_temp>(0.2*test_loss+0.8*(re_loss)):

                loss_temp=0.2*test_loss+0.8*(re_loss)
                checkpoint={'epoch':epoch,'model':discriminator.state_dict(),'optimizer':optimizer_D.state_dict()}
                torch.save(checkpoint,'discribe')
    