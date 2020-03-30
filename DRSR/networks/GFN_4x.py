import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os
import torch.nn.functional as F
from networks.net import ConvLayer, UpsampleConvLayer, ResidualBlock

class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 5, stride, 2, bias=True),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 5, stride, 2, bias=True),
            nn.InstanceNorm2d(outchannel)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _ResBlockSR(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlockSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.InstanceNorm2d(outchannel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True),
            nn.InstanceNorm2d(outchannel)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _EDNet(nn.Module):
    def __init__(self, res_blocks=18):
        super(_EDNet, self).__init__()

       # rgb_mean = (0.5204, 0.5167, 0.5129)
       # self.sub_mean = MeanShift(1., rgb_mean, -1)
       # self.add_mean = MeanShift(1., rgb_mean, 1)

        self.conv_input = ConvLayer(3, 64, kernel_size=11, stride=1)
        self.conv2x = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.conv4x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv8x = ConvLayer(128, 256, kernel_size=3, stride=2)
       # self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)


        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        #self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.convd8x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.convd2x = UpsampleConvLayer(64, 64, kernel_size=3, stride=2)

        self.conv_output = ConvLayer(64, 3, kernel_size=3, stride=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.conv_input(x))
        res2x = self.relu(self.conv2x(x))
        res4x = self.relu(self.conv4x(res2x))

        res8x = self.relu(self.conv8x(res4x))
        #res16x = self.relu(self.conv16x(res8x))

        res_dehaze = res8x
        res8x = self.dehaze(res8x)
        res8x = torch.add(res_dehaze, res8x)

        #res16x = self.relu(self.convd16x(res16x))
        #res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        #res8x = torch.add(res16x, res8x)

        res8x = self.relu(self.convd8x(res8x))
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)

        res4x = self.relu(self.convd4x(res4x))
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.relu(self.convd2x(res2x))
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        
        res_ft = res2x
        x = torch.add(res2x, x)
        x = self.conv_output(x)

        return res_ft, x


class _SRMoudle(nn.Module):
    def __init__(self):
        super(_SRMoudle, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), 1, padding=3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock = self._makelayers(64, 64, 6, 1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBlockSR(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        res1 = self.resBlock(con1)
        con2 = self.conv2(res1)
        sr_feature = torch.add(con2, con1)
        return sr_feature

class _GateMoudle(nn.Module):
    def __init__(self):
        super(_GateMoudle, self).__init__()

        self.conv1 = nn.Conv2d(131,  64, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.conv2(con1)
        return scoremap

class _ReconstructMoudle(nn.Module):
    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
        self.resBlock = self._makelayers(64, 64, 6)
        self.conv1 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64, 3, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1))
        con2 = self.conv2(pixelshuffle1)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))
        con3 = self.relu3(self.conv3(pixelshuffle2))
        sr_deblur = self.conv4(con3)
        return sr_deblur

class UpLayer(nn.Module):
    def __init__(self):
        super(UpLayer, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            #nn.PReLU(),
            nn.LeakyReLU(0.2),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        #self.up = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        h = self.up(x)
        return h
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurMoudle      = self._make_net(_EDNet)
        self.srMoudle          = self._make_net(_SRMoudle)
        self.geteMoudle        = self._make_net(_GateMoudle)
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)
        self.uplayer = self.make_layer(UpLayer, 1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
        
    def forward(self, x, gated, isTest):
        if isTest == True:
            origin_size = x.size()
            input_size  = (math.ceil(origin_size[2]/4)*4, math.ceil(origin_size[3]/4)*4)
            out_size    = (origin_size[2]*4, origin_size[3]*4)
            x           = nn.functional.upsample(x, size=input_size, mode='bilinear')
            
        deblur_feature, deblur_out = self.deblurMoudle(x)
        sr_feature = self.srMoudle(x)
        if gated == True:
            scoremap1 = self.geteMoudle(torch.cat((deblur_feature, x, sr_feature), 1))
            repair_feature = torch.mul(scoremap1, deblur_feature)
            fusion_feature1 = torch.add(sr_feature, repair_feature)
            scoremap2 = self.geteMoudle(torch.cat((deblur_feature, x, fusion_feature1), 1))
            repair_feature = torch.mul(scoremap2, deblur_feature)
            fusion_feature2 = torch.add(fusion_feature1, repair_feature)
            scoremap3 = self.geteMoudle(torch.cat((deblur_feature, x, fusion_feature2), 1))
            repair_feature = torch.mul(scoremap3, deblur_feature)
            fusion_feature = torch.add(fusion_feature2, repair_feature)

        else:
            scoremap = torch.cuda.FloatTensor().resize_(sr_feature.shape).zero_()+1
            repair_feature = torch.mul(scoremap, deblur_feature)
            fusion_feature = torch.add(sr_feature, repair_feature)


        recon_out = self.reconstructMoudle(fusion_feature)

        if isTest == True:
            recon_out = nn.functional.upsample(recon_out, size=out_size, mode='bilinear')

        SR_UP = self.uplayer(x)
        recon_out = recon_out + SR_UP
        return deblur_out, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)




