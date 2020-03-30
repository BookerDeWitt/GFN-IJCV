import torch
import torch.nn as nn
import math
import torch.nn.init as init
from networks.net import ConvLayer, UpsampleConvLayer, ResidualBlock
import torch.nn.functional as F

class _Residual_Block_SR(nn.Module):
    def __init__(self):
        super(_Residual_Block_SR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output
        
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

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



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual1 = self.make_layer(_Residual_Block_SR, 8)
        self.residual2 = self.make_layer(_Residual_Block_SR, 8)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)
        self.conv_output = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),  # eliminate the artifacts
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        )

        self.ednet = self.make_layer(_EDNet, 1)
        self.conv_channel = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            )


        ##############Attention Model#####################
        self.attention_model = nn.Sequential(
            nn.Conv2d(in_channels=131, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        [deblur_feature, lr_deblur] = self.ednet(x)
      #  deblur_feature = self.conv_channel(deblur_feature)

        out = self.conv_input(x)
        residual = out
        out = self.residual1(out)
        out = self.conv_mid(out)
        out = torch.add(out, residual)
        sr_feature = out
        scoremap1 = self.attention_model(torch.cat((deblur_feature, x, sr_feature), 1))
        repair_feature = torch.mul(scoremap1, deblur_feature)
        fusion_feature1 = torch.add(sr_feature, repair_feature)
        scoremap2 = self.attention_model(torch.cat((deblur_feature, x, fusion_feature1), 1))
        repair_feature = torch.mul(scoremap2, deblur_feature)
        fusion_feature2 = torch.add(fusion_feature1, repair_feature)
        scoremap3 = self.attention_model(torch.cat((deblur_feature, x, fusion_feature2), 1))
        repair_feature = torch.mul(scoremap3, deblur_feature)
        fusion_feature = torch.add(fusion_feature2, repair_feature)
        #att_in = torch.cat((x,att_in1, att_in2), dim=1)
        #scoremap = self.attention_model(att_in)
        #detail_ft = torch.mul(att_in1, scoremap)
        #out = torch.add(out, att_in1)

        out = self.residual2(fusion_feature)
        out = self.upscale4x(out)
        sr = self.conv_output(out)
        return lr_deblur,  sr