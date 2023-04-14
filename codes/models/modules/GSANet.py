from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

## DRDB with dilation dense
class make_dilation_dense(nn.Module):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class DRDB(nn.Module):

    def __init__(self, nChannels, denseLayer, growthRate):
        super(DRDB, self).__init__()
        num_channels = nChannels
        modules = []
        for i in range(denseLayer):
            modules.append(make_dilation_dense(num_channels, growthRate))
            num_channels += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,padding=1,groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
    )

class SpatialAttentionModule(nn.Module):

    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        # self.att1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, bias=True)
        self.att1 = SeparableConvolution(n_feats * 2, n_feats * 2)
        self.att2 = SeparableConvolution(n_feats * 2, n_feats)
        # self.att2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class UnetSmall(nn.Module):
    def __init__(self, mid_channels):
        super(UnetSmall, self).__init__()

        self.mid_channels = mid_channels

        self.conv1 = nn.Conv2d(3, self.mid_channels, kernel_size=3, stride=1, padding=1)
        ## Denoise
        # down
        self.downsample = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            SeparableConvolution(self.mid_channels * 2, self.mid_channels * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # up
        self.upsample = nn.Sequential(
            SeparableConvolution(self.mid_channels * 2, self.mid_channels * 4),
            nn.PixelShuffle(2)
        )
        self.conv2 = nn.Conv2d(self.mid_channels, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return x
    

class Model_G(nn.Module):

    def __init__(self, mid_channels, align_version='v0'):
        super(Model_G, self).__init__()
    
        self.mid_channels = mid_channels

        ## Denoise
        self.denoise = UnetSmall(16)

        ## Spatial attention module
        self.att_module_l = SpatialAttentionModule(self.mid_channels)
        self.att_module_h = SpatialAttentionModule(self.mid_channels)

        ## feature extraction
        self.feat_exract = nn.Sequential(
            nn.Conv2d(3, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        ## conv
        self.conv1 = SeparableConvolution(224, 32)
        # self.conv2 = nn.Conv2d(32, mid_channels, kernel_size=3, padding=1, bias=True)
        ## channel attention
        self.channel_att = nn.Sequential(
            # SeparableConvolution(224, 224),
            # nn.Conv2d(224, mid_channels, kernel_size=3, padding=1, bias=True),
            CAB(n_feat=mid_channels, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        )
       
        self.conv2 = SeparableConvolution(32, 32)

        # DRDB useful
        self.RDB = DRDB(self.mid_channels, 3, 16)

        # post conv
        self.post_conv = nn.Sequential(
            SeparableConvolution(self.mid_channels, self.mid_channels),
            # nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.mid_channels, 3, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.LeakyReLU(inplace=True)


    def LDR2HDR(self, img, float_expo, gamma=2.24):
        '''Map the LDR inputs into the HDR domain'''
        img = img.permute(1,2,3,0)  # ????
        # exp_img = img ** gamma / expo
        exp_img = (((img**gamma)*2.0**(-1*float_expo))**(1/gamma))
        exp_img = exp_img.permute(3,0,1,2)
        return exp_img

    def forward(self, X, exposure_values):

        # for a single test scene, the input tensor X has shape (1, 3, 3, 1060, 1900) - (batch_size, num images, channels, height, width) 
        x1_t = X[:,0,:,:,:]
        x2_t = X[:,1,:,:,:]
        x3_t = X[:,2,:,:,:]

        # map the LDR inputs into the HDR domain
        x1_l = self.LDR2HDR(x1_t, exposure_values[:,0])
        x2_l = self.LDR2HDR(x2_t, exposure_values[:,1])
        x3_l = self.LDR2HDR(x3_t, exposure_values[:,2])
       
        ### Attention Network
        ## 1 group: x1_t, x1_l, x2_t  
        y1_1 = self.feat_exract(x1_t)
        y1_2 = self.feat_exract(self.denoise(x1_l))
        y1_3 = self.feat_exract(x2_t)
        ## attention short
        y1_ = self.att_module_l(y1_1, y1_2)
        f1 = y1_1 * y1_
        y2_ = self.att_module_h(y1_2, y1_3)
        f2 = y1_3 * y2_
       
        y1 = torch.cat((f1, f2), 1)

        ## 2 group: x1_l, x2_l, x3_l add details
        y2_1 = self.feat_exract(self.denoise(x1_l))
        y2_2 = self.feat_exract((x2_l))
        y2_3 = self.feat_exract(x3_l)

        y2 = torch.cat((y2_1, y2_2, y2_3), 1)

        ## 3 group: x2_t, x3_l, x3_t
        y3_1 = self.feat_exract(x2_t)
        y3_2 = self.feat_exract(x3_l)
        y3_3 = self.feat_exract(x3_t)
        ## attention long
        y3_ = self.att_module_l(y3_1, y3_2)
        f3 = y3_1 * y3_
        y4_ = self.att_module_h(y3_2, y3_3)
        f4 = y3_3 * y4_
        y3 = torch.cat((f3,f4), 1)
        
        ### Fusion Network
        y = torch.cat((y1, y2, y3), 1) 
        y = self.conv1(y)
        y = self.channel_att(y)
        y = self.conv2(y)

        ## RDB
        y = self.RDB(y)
        y = self.post_conv(y)

        return y

