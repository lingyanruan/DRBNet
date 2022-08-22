'''
This source code is licensed under the license found in the LICENSE file.
This is the implementation of the "Learning to deblur using light field generated and real defocus images" paper accepted to CVPR 2022. 
Project GitHub repository: https://github.com/lingyanruan/DRBNet
Email: lyruanruan@gmail.com
Copyright (c) 2022-present, Lingyan Ruan
'''


import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from pathlib import Path
import cv2
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, act='LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1,inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True, res_num=1):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias, res_num=res_num)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, res_num):
        super(ResnetBlock, self).__init__()
        self.res_num = res_num
        self.stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            ) for i in range(res_num)
        ])
    def forward(self, x):

        if self.res_num > 1:
            temp = x

        for i in range(self.res_num):
            xx = self.stem[i](x)
            x = x + xx
        if self.res_num > 1:
            x = x + temp

        return x

def FAC(feat_in, kernel, ksize):
    """
    customized FAC
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    if channels ==3 and kernels == ksize*ksize:
        ####
        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize)
        kernel = torch.cat([kernel,kernel,kernel],channels)
        kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 

    else:
        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize) 
        kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1) 
                
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        
    return feat_out

class DRBNet_single(nn.Module):
    def __init__(self, ):
        super(DRBNet_single, self).__init__()


        ks = 3 
        
        ch1 = 32
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8
        self.ch4 = ch4
        self.kernel_width = 7
        self.kernel_dim = self.kernel_width*self.kernel_width

  
        # feature extractor
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)
        # here has a dynamic filter and res

        self.img_d8_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch4, kernel_size=ks, stride=1) 
        )

        self.upconv3_kernel = nn.Sequential(
            conv(ch4*2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) 

        self.upconv3_res = nn.Sequential(
            conv(ch4*2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)            
        )
        
        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)
       

        self.img_d4_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1), 
        )


        self.upconv2_kernel = nn.Sequential(
            conv(ch3*2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) 

        self.upconv2_res = nn.Sequential(
            conv(ch3*2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)            
        )
        self.img_d2_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1) 
        )

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)
        

        self.img_d1_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch1, kernel_size=ks, stride=1), 
        )


        self.upconv1_kernel = nn.Sequential(
            conv(ch2*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) 

        self.upconv1_res = nn.Sequential(
            conv(ch2*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)           
        )


        self.upconv0_kernel = nn.Sequential(
            conv(ch1*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) 

        self.upconv0_res = nn.Sequential(
            conv(ch1*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)           
        )

##########################################################################
    def forward(self, C):
    # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C))) 
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1))) 
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2))) 
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3))) 

        f = self.conv4_4(f_C)

        img_d8 = F.interpolate(C, scale_factor=1/8, mode='area')
        img_d8_feature = self.img_d8_feature(img_d8)
        feature_d8 = torch.cat([f,img_d8_feature],1) #ch4*2
        kernel_d8 = self.upconv3_kernel(feature_d8)

        res_f8 = self.upconv3_res(feature_d8)
        
        est_img_d8 = img_d8 + FAC(img_d8, kernel_d8, self.kernel_width) + res_f8

        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        est_img_d4_interpolate =F.interpolate(est_img_d8, scale_factor=2, mode='area')

        
        img_d4_feature = self.img_d4_feature(est_img_d4_interpolate)
        feature_d4 = torch.cat([f,img_d4_feature],1) 
        kernel_d4 = self.upconv2_kernel(feature_d4)

        res_f4 = self.upconv2_res(feature_d4)

        est_img_d4 = est_img_d4_interpolate + FAC(est_img_d4_interpolate, kernel_d4, self.kernel_width) + res_f4

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))


        est_img_d2_interpolate =F.interpolate(est_img_d4, scale_factor=2, mode='area')
    
        img_d2_feature = self.img_d2_feature(est_img_d2_interpolate)
        feature_d2 = torch.cat([f,img_d2_feature],1) 
        
        kernel_d2 = self.upconv1_kernel(feature_d2) 
        res_f2 = self.upconv1_res(feature_d2)

        est_img_d2 = est_img_d2_interpolate + FAC(est_img_d2_interpolate, kernel_d2, self.kernel_width) + res_f2


        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        est_img_d1_interploate =F.interpolate(est_img_d2, scale_factor=2, mode='area')
    
        img_d1_feature = self.img_d1_feature(est_img_d1_interploate)
        feature_d1 = torch.cat([f,img_d1_feature],1)
        kernel_d1 = self.upconv0_kernel(feature_d1)
        
        res_f1 = self.upconv0_res(feature_d1)

        est_img_d1 = est_img_d1_interploate + FAC(est_img_d1_interploate, kernel_d1,self.kernel_width) + res_f1

        est_img_d1_ = torch.clip(est_img_d1,-1.0,1.0)
   
        return est_img_d1_



##########################################################################################
## dual views net


class DeblurNet_dual(nn.Module):
    def __init__(self,):
        super(DeblurNet_dual, self).__init__()
       


        ks = 3 

        ch1 = 32
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8
        self.ch4 = ch4
        self.kernel_width = 7
        self.kernel_dim = self.kernel_width*self.kernel_width

        # feature extractor
        self.conv1_1 = conv(6, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            conv(ch4, ch4, kernel_size=ks))


        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)
       

        self.img_d8_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch4, kernel_size=ks, stride=1) 
        )

        self.upconv3_kernel = nn.Sequential(
            conv(ch4*2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) 

        self.upconv3_res = nn.Sequential(
            conv(ch4*2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)            
        )
        
        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)
        

        self.img_d4_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1), 
        )


        self.upconv2_kernel = nn.Sequential(
            conv(ch3*2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) 

        self.upconv2_res = nn.Sequential(
            conv(ch3*2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)            
        )
        self.img_d2_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1) 
        )

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)
        

        self.img_d1_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch1, kernel_size=ks, stride=1), 
        )


        self.upconv1_kernel = nn.Sequential(
            conv(ch2*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) # 5*5 kernel

        self.upconv1_res = nn.Sequential(
            conv(ch2*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)           
        )


        self.upconv0_kernel = nn.Sequential(
            conv(ch1*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1,act=None)
        ) # 5*5 kernel

        self.upconv0_res = nn.Sequential(
            conv(ch1*2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)           
        )



##########################################################################
    def forward(self, C,R,L):

        # feature extractor
       
        input = torch.cat([R,L],1)
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(input))) 
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1))) 
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2))) 
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3))) 

        f = self.conv4_4(f_C)

        img_d8 = F.interpolate(C, scale_factor=1/8, mode='area')
        img_d8_feature = self.img_d8_feature(img_d8)
        feature_d8 = torch.cat([f,img_d8_feature],1) 
        kernel_d8 = self.upconv3_kernel(feature_d8)

        res_f8 = self.upconv3_res(feature_d8)
        
        est_img_d8 = img_d8 + FAC(img_d8, kernel_d8, self.kernel_width) + res_f8

     
        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        est_img_d4_interpolate =F.interpolate(est_img_d8, scale_factor=2, mode='area')

        
        img_d4_feature = self.img_d4_feature(est_img_d4_interpolate)
        feature_d4 = torch.cat([f,img_d4_feature],1) 
        kernel_d4 = self.upconv2_kernel(feature_d4)

        res_f4 = self.upconv2_res(feature_d4)

        est_img_d4 = est_img_d4_interpolate + FAC(est_img_d4_interpolate, kernel_d4, self.kernel_width) + res_f4


        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))


        est_img_d2_interpolate =F.interpolate(est_img_d4, scale_factor=2, mode='area')
    
        img_d2_feature = self.img_d2_feature(est_img_d2_interpolate)
        feature_d2 = torch.cat([f,img_d2_feature],1) 
        
        kernel_d2 = self.upconv1_kernel(feature_d2) 
        res_f2 = self.upconv1_res(feature_d2)

        est_img_d2 = est_img_d2_interpolate + FAC(est_img_d2_interpolate, kernel_d2, self.kernel_width) + res_f2


        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        est_img_d1_interploate =F.interpolate(est_img_d2, scale_factor=2, mode='area')
    
        img_d1_feature = self.img_d1_feature(est_img_d1_interploate)
        feature_d1 = torch.cat([f,img_d1_feature],1)
        kernel_d1 = self.upconv0_kernel(feature_d1)
        
        res_f1 = self.upconv0_res(feature_d1)

        est_img_d1 = est_img_d1_interploate + FAC(est_img_d1_interploate, kernel_d1,self.kernel_width) + res_f1
        est_img_d1_ =torch.clip(est_img_d1,-1.0,1.0)
 
        return est_img_d1_










