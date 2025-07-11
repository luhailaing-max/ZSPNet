
# -*- coding: utf-8 -*-
"""

"""
from skimage import io
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import  nn, einsum
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import math

import torch.optim as optim
import warnings
import unfoldNd

from einops import rearrange
from enum import Enum



class Mlp2(nn.Module):
    def __init__(self, in_features,out_features=None, mlpratio = 2,act_layer=nn.GELU, drop=0.):
        super().__init__()

        if in_features == 1:
            self.fc1 = nn.Linear(1, out_features//2)
            self.fc2 = nn.Linear(out_features//2, out_features)
        else:
            self.fc1 = nn.Linear(in_features, in_features*mlpratio)
            self.fc2 = nn.Linear(in_features*mlpratio, out_features)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


##############################################################################
#####################Features Fusion Block#################################### FFB

class CNN3D(nn.Module):
    def __init__(self,indim, outdim):
        super(CNN3D, self).__init__()

        self.cnn3d1= nn.Conv3d(indim, indim*2, kernel_size=3, padding=1)
        # self.cnn3dbn = nn.BatchNorm3d(indim*2)
        self.cnn3drelu = nn.ReLU(inplace=True)
        self.cnn3d2= nn.Conv3d(indim*2,outdim, kernel_size=3, padding=1)

    def forward(self, input):
        # B,N,C = input.shape
        # SIZE = int(np.sqrt(N))
        # input = input.transpose(1, 2).view(B, C, SIZE, SIZE)
        # input = input.unsqueeze(2)
        x = self.cnn3d1(input)
        # x = self.cnn3dbn(x)
        x = self.cnn3drelu(x)
        x = self.cnn3d2(x)    
        # x = x.squeeze()
        # x = x.flatten(2).transpose(1, 2)
        return x


class Features_Fusion_Modle(nn.Module):
    def __init__(self,in_chans, out_chans,num = 2):
        super(Features_Fusion_Modle, self).__init__()

        self.mslayer = Mlp2(in_chans,out_chans)
        self.panlayer= Mlp2(1,out_chans)
        # self.fusionlayer = nn.ModuleList()
        # for i_ in range(num):
        #     layer = CNN3D(out_chans,out_chans)
        #     self.fusionlayer.append(layer)
        self.cnn3dlayer1 = CNN3D(out_chans,out_chans)
        self.maxpool3dlayer = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(1, 1, 1))
        self.cnn3dlayer2 = CNN3D(out_chans,out_chans)

    def forward(self, msf, pan):

        if len(msf.shape) == 4:
            msf = msf.flatten(2).transpose(1, 2)        
        x_ms = self.mslayer(msf)
        pan = pan.flatten(2).transpose(1, 2)
        x_pan = self.panlayer(pan)

        B,N,C = x_ms.shape
        SIZE = int(np.sqrt(N))
        x_ms = x_ms.transpose(1, 2).view(B, C, SIZE, SIZE)

        B,N,C = x_pan.shape
        SIZE = int(np.sqrt(N))
        x_pan = x_pan.transpose(1, 2).view(B, C, SIZE, SIZE)


        x_ms = x_ms.unsqueeze(2)
        x_pan = x_pan.unsqueeze(2)
        x = torch.cat((x_ms, x_pan), dim=2)

        x = self.cnn3dlayer2(x)
        x = self.maxpool3dlayer(x)
        x = self.cnn3dlayer1(x)
        x = x.squeeze()

        return x

##############################################################################
#####################Spatial Information Reconstruction Block######################## SIRB

class Spatial_Attention(nn.Module):
    """
    Tips:
        From SKNet (https://github.com/implus/SKNet)
    """
    def __init__(self, dim):
        super().__init__()
        # K = d*(k_size-1)+1
        # (H - k_size + 2padding)/stride + 1
        # (5,1)-->(7,3)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # K=5, 64-5+4+1=64
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        # (3,1)-->(5,2)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  #
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) # K=9, 64-9+8 + 1

        # (5,1)-->(7,4)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  #
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=12, groups=dim, dilation=4) # K=25, 64-25+2*12 + 1

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn



class UP_Layer(nn.Module):
    def __init__(self, in_s, out_s): # Image size
        super().__init__()

        act_layer=nn.GELU
        self.fc1 = nn.Linear(in_s, in_s*2)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_s*2, out_s)

    def forward(self, x):
        x = x.flatten(2)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        B,C,N = x.shape
        SIZE = int(np.sqrt(N))
        x = x.view(B, C, SIZE, SIZE)
        return x

class Information_Reconstruction_Module(nn.Module):
    def __init__(self, in_s, out_s, dim):
        super().__init__()
        self.satt=Spatial_Attention(dim)
        self.uplayer = UP_Layer(in_s*in_s, out_s*out_s )

    def forward(self, x):
    
        x = self.satt(x)
        x = self.uplayer(x)

        return x


##############################################################################
##################### Multi Scale Fusion Block########################MSFB
class Channel_Attention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(Channel_Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class Multi_Scale_Feature_Extraction(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, groups=in_c, bias=True)
        self.relu3 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_c, out_c,kernel_size=5, stride=1, padding=2, groups=in_c, bias=True)
        self.relu5 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3, groups=in_c, bias=True)
        self.relu7 = nn.ReLU()

        self.catt = Channel_Attention(out_c*3)
        self.conv = nn.Conv2d(out_c*3, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
    
        x1 =  self.relu3(self.conv3(x))
        x2 =  self.relu5(self.conv5(x))
        x3 =  self.relu7(self.conv7(x))
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.catt(x)
        x = self.conv(x)

        return x

##############################################################################
##################### others ########################
class Fist_layer(nn.Module):

    def __init__(self, in_chans=3, out_chans=96):
        super().__init__()
        self.proj = Mlp2(in_chans,out_chans)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        return x



class Out_layer(nn.Module):

    def __init__(self, in_chans=96, out_chans=6):
        super().__init__()

        self.proj = Mlp2(in_chans, out_chans)
        self.outchannel = out_chans

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        B,N,C = x.shape
        SIZE = int(np.sqrt(N))
        out = x.transpose(1, 2).view(B, C, SIZE, SIZE)
        return out



class Net(nn.Module):
    def __init__(self, inc, outc,fetures,dim=64,mlist = [1,1,1]):
        super(Net, self).__init__()
        
        self.inchannel = inc
        self.outchannel = outc

        self.fistlayer = Fist_layer(self.inchannel,dim)

        self.FFM = Features_Fusion_Modle(dim,dim)
        self.IRM1 = Information_Reconstruction_Module(fetures[0],fetures[1],dim)
        # self.FFM2 = Features_Fusion_Modle(dim,dim)
        self.IRM2 = Information_Reconstruction_Module(fetures[1],fetures[2],dim)
        # self.FFM3 = Features_Fusion_Modle(dim,dim)

        self.MSFE = Multi_Scale_Feature_Extraction(dim,dim)

        self.out_layer = Out_layer(dim,outc)

    def forward(self,inp,pan,pan1,pan2):
        
        msf = self.fistlayer(inp)
      
        x = self.FFM(msf, pan2)
        # x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = self.IRM1(x)

        x = self.FFM(x, pan1)
        # x = F.upsample(x, scale_factor=2, mode='bilinear')        
        x = self.IRM2(x)

        x = self.FFM(x, pan)

        x = self.MSFE(x)

        out3 = self.out_layer(x)


        return None,None,out3

from Pan_mtfresize import interp23tap_torch
class modelv20(nn.Module):
    def __init__(self,inc,outc,fetures=[4,8,16],dim=64,mlist = [1,1,1,1]):
        super(modelv20, self).__init__()

        self.n1 = Net(inc,outc,fetures,dim, mlist)

        self.weights = torch.nn.Parameter(torch.ones(2).float())
    def forward(self,inp,pan,pan1,pan2,mslr2,mslr4):

        out1,out2,out3 = self.n1(inp,pan,pan1,pan2)

        out3 = out3 + mslr4

        return out1,out2,out3

