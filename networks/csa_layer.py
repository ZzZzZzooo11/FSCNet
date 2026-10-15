import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(nn.Module):

    def __init__(self, in_channels, channels, kernel_size=3, stride=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, norm_layer=None, **kwargs):
        super(SplAtConv2d, self).__init__()

        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        # 分支卷积
        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                              groups=groups * radix, bias=bias, **kwargs)

        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
            self.bn2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3,
                      groups=channels, bias=bias),
            norm_layer(channels) if self.use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=bias)
        )

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn: x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        x1, x2 = torch.split(x, rchannel // self.radix, dim=1)

        # 特征融合
        x_gap = x1 + x2
        x_gap = self.conv2(x_gap)
        if self.use_bn: x_gap = self.bn2(x_gap)
        x_gap = self.relu(x_gap)

        # 注意力计算
        gap = F.adaptive_avg_pool2d(x_gap, 1)
        gap = self.fc1(gap)
        if self.use_bn: gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        attens = torch.split(atten, rchannel // self.radix, dim=1)

        out = sum([att * split for (att, split) in zip(attens, (x1, x2))])
        return out.contiguous()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=True,
                 norm_layer=None, custom=0):
        super(Bottleneck, self).__init__()

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        if custom != 0:
            inplanes = custom

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)

        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.conv2 = SplAtConv2d(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=dilation, groups=cardinality, bias=False,
            radix=radix, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, radix=2, groups=1, bottleneck_width=64,
                 stem_width=64, avg_down=True, avd=True, avd_first=False,
                 norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2
        self.avg_down = avg_down
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        sw = stem_width
        ConvFea = [sw, sw * 2, sw * 4, sw * 8, sw * 16]

        self.layer1 = self._make_layer(block, ConvFea[0], layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, ConvFea[1], layers[1], stride=1, norm_layer=norm_layer, is_first=True)
        self.layer3 = self._make_layer(block, ConvFea[2], layers[2], stride=1, norm_layer=norm_layer, is_first=True)
        self.layer4 = self._make_layer(block, ConvFea[2], layers[3], stride=1, norm_layer=norm_layer, is_first=True)

        # Decoder Layers - 拼接后的通道匹配
        self.layer5 = self._make_layer(block, ConvFea[1], layers[0], stride=1, norm_layer=norm_layer,
                                       inchannel=ConvFea[3] * 4, is_first=True)
        self.layer6 = self._make_layer(block, ConvFea[0], layers[1], stride=1, norm_layer=norm_layer,
                                       inchannel=ConvFea[2] * 4, is_first=True)
        self.layer7 = self._make_layer(block, ConvFea[0] // 2, layers[2], stride=1, norm_layer=norm_layer,
                                       inchannel=ConvFea[1] * 4, is_first=True)
        self.layer8 = self._make_layer(block, ConvFea[0] // 2, layers[3], norm_layer=norm_layer,
                                       inchannel=ConvFea[0] * 4, is_first=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, inchannel=0, is_first=True):
        downsample = None
        if inchannel != 0:
            self.inplanes = inchannel

        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                down_layers.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                down_layers.append(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            radix=self.radix, cardinality=self.cardinality,
                            bottleneck_width=self.bottleneck_width,
                            avd=self.avd, avd_first=self.avd_first,
                            dilation=1, is_first=is_first,
                            norm_layer=norm_layer, custom=inchannel))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)


def CSA(base_c=64, **kwargs):
    calculated_stem_width = base_c // 2
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1,
                   bottleneck_width=calculated_stem_width,
                   stem_width=calculated_stem_width,
                   avg_down=True, avd=True, **kwargs)
    return model