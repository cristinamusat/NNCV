import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

N_CLASSES = 19

# U-Net Implementation

class UNet(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, in_channels=3, n_classes=N_CLASSES):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

    """
    DeepLabv3_plus
    Based on the architecture from the original paper:
    Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    ArXiv. https://arxiv.org/pdf/1802.02611.pdf
    References: jfzhang95. (2018). PyTorch DeepLab-XCeption. GitHub. https://github.com/jfzhang95/pytorch-deeplab-xception
    """     

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

def ResNet34(nInputChannels=3, os=16, pretrained=False):
    return ResNet(nInputChannels, BasicBlock, [3, 4, 6, 3], os, pretrained)

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=N_CLASSES, os=16, pretrained=False):
        super(DeepLabv3_plus, self).__init__()
        self.resnet_features = ResNet34(nInputChannels, os, pretrained)
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPP_module(512, 64, rate=rates[0])
        self.aspp2 = ASPP_module(512, 64, rate=rates[1])
        self.aspp3 = ASPP_module(512, 64, rate=rates[2])
        self.aspp4 = ASPP_module(512, 64, rate=rates[3])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 64, 1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(320, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.last_conv = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        )
    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                   int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = F.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

# FCN
# Source: https://github.com/kerrgarr/SemanticSegmentationCityscapes.git
# Author: kerrgarr

def down_conv(small_channels, big_channels, pad):
    return nn.Sequential(
        nn.Conv2d(small_channels, big_channels, 3, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(big_channels),
        nn.Conv2d(big_channels, big_channels, 3, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(big_channels)
    )

def up_conv(big_channels, small_channels, pad):
    return nn.Sequential(
        nn.Conv2d(big_channels, small_channels, 3, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(small_channels),
        nn.Conv2d(small_channels, small_channels, 3, padding=pad),
        nn.ReLU(),
        nn.BatchNorm2d(small_channels)
    )

class my_FCN(nn.Module):
    def crop(self, a, b):
        Ha = a.size()[2]
        Wa = a.size()[3]
        adapt = nn.AdaptiveMaxPool2d((Ha, Wa))
        crop_b = adapt(b)
        return crop_b    
   
    def __init__(self):
        super().__init__()
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)         
        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.std = torch.Tensor([0.25, 0.25, 0.25])
        a = 32
        b = a*2  # 64
        c = b*2  # 128
        d = c*2  # 256
        n_class = N_CLASSES
        self.conv_down1 = down_conv(3, a, 1)
        self.conv_down2 = down_conv(a, b, 1)
        self.conv_down3 = down_conv(b, c, 1)
        self.conv_down4 = down_conv(c, d, 1)
        self.bottleneck = nn.ConvTranspose2d(d, c, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.conv_up3 = up_conv(c, b, 1)
        self.upsample3 = nn.ConvTranspose2d(b, a, kernel_size=3, stride=2, padding=1, output_padding=1)   
        self.classifier = nn.Conv2d(a, n_class, kernel_size=1) 
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        z = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)
        conv1 = self.conv_down1(z)
        mx1 = self.maxpool(conv1)
        conv2 = self.conv_down2(mx1)
        mx2 = self.maxpool(conv2)
        conv3 = self.conv_down3(mx2)
        mx3 = self.maxpool(conv3)
        conv4 = self.conv_down4(conv3)
        score = self.bottleneck(conv4)
        crop_conv3 = self.crop(score, conv3)
        score = score + crop_conv3
        score = self.conv_up3(score)
        score = self.upsample3(score)
        crop_conv1 = self.crop(score, conv1)
        score = score + crop_conv1           
        score = self.classifier(score)
        out = nn.functional.interpolate(score, size=(H, W))
        return out

models = {
    'my_fcn': my_FCN,
    'unet': UNet,
    'deeplab': DeepLabv3_plus
}