#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: work5_20201215
@Author: sol@JinGroup
@File: hehe_resnet.py
@Time: 1/27/21 1:51 PM
@E-mail: hesuozhang@gmail.com
'''

"""This script comes from the official resnet implementation of torchvision.

We add the DCN switch(from torchvision.ops) and remove some unnecessary codes.
"""
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import DeformConv2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    """Basic Residual Block.

    Args:
        inplanes (int): channels of input tensor.
        planes (int): channels of output tensor.
        stride (int or tuple, optional): Stride of this block. Default: 1
        downsample (nn.Module): downsample module. Default: None
        groups (int, optional): groups param for conv layer. Default: 1.
        base_width (int, optional): base_width param for ResBlock. Defaults to 64.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        norm_layer (nn.Module): batch norm module. Default: None
        with_dcn (bool, optional): dcn switch. Defaults to False.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, with_dcn=False):
        """Init function."""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.with_dcn = with_dcn
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if self.with_dcn:
            offset_channels = 18
            self.conv1_offset = nn.Conv2d(inplanes, groups * offset_channels, kernel_size=3, stride=stride, padding=1)
            self.conv1_offset.weight.data.fill_(0)
            self.conv1_offset.bias.data.fill_(0)

            self.conv1 = DeformConv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if self.with_dcn:
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(planes, groups * offset_channels, kernel_size=3, padding=1)
            self.conv2_offset.weight.data.fill_(0)
            self.conv2_offset.bias.data.fill_(0)

            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward function."""
        identity = x
        if self.with_dcn:
            offset = self.conv1_offset(x)
            out = self.conv1(x, offset)
        else:
            out = self.conv1(x)
        # out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.with_dcn:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck Residual Block.

    Args:
        inplanes (int): channels of input tensor.
        planes (int): channels of output tensor.
        stride (int or tuple, optional): Stride of this block. Default: 1
        downsample (nn.Module): downsample module. Default: None
        groups (int, optional): groups param for conv layer. Default: 1.
        base_width (int, optional): base_width param for ResBlock. Defaults to 64.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        norm_layer (nn.Module): batch norm module. Default: None
        with_dcn (bool, optional): dcn switch. Defaults to False.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, with_dcn=False):
        """Init function."""
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.with_dcn = with_dcn
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.with_dcn:
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(
                width, groups * offset_channels, kernel_size=3, stride=stride, padding=1
            )
            self.conv2_offset.weight.data.fill_(0)
            self.conv2_offset.bias.data.fill_(0)

            self.conv2 = DeformConv2d(
                width, width, kernel_size=3, stride=stride, padding=1,
                groups=groups, dilation=dilation, bias=False
            )
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward function."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.with_dcn:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        block (nn.Module): Block type, one of [BasicBlock, BottleNeck].
        layers (list[int]): Specify the number of block in each conv stage.
        deep_stem (bool, optional): Replace 7x7 conv in input stem with 3 3x3 conv. Defaults to False.
            refer to paper: <Bag of Tricks for Image Classification with Convolutional Neural Networks>
        avg_down (bool, optional): Use AvgPool instead of stride conv when downsampling in the bottleneck.
            Defaults to False.
        groups (int, optional): Param for group conv. Defaults to 1.
        width_per_group (int, optional): Param for group conv. Defaults to 64.
        norm_layer (nn.Module, optional): Normalize layer. Defaults to None.
        stage_with_dcn (list or None, optional): Switch for DCN. Defaults to None.
        arch (str, optional): Name of backbone. Defaults to None.
    """

    def __init__(
        self, block, layers, deep_stem=False, avg_down=False, groups=1,
        width_per_group=64, norm_layer=None, stage_with_dcn=None, arch=None
    ):
        """Init function."""
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.stage_with_dcn = stage_with_dcn if stage_with_dcn is not None else [False] * 4
        assert len(self.stage_with_dcn) == 4

        assert deep_stem == avg_down
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_dcn=self.stage_with_dcn[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, with_dcn=self.stage_with_dcn[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, with_dcn=self.stage_with_dcn[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, with_dcn=self.stage_with_dcn[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # init weight/bias of dcn offset layer to 0
        for m in self.modules():
            if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                if hasattr(m, 'conv1_offset'):
                    m.conv1_offset.weight.data.fill_(0)
                    m.conv1_offset.bias.data.fill_(0)
                if hasattr(m, 'conv2_offset'):
                    m.conv2_offset.weight.data.fill_(0)
                    m.conv2_offset.bias.data.fill_(0)

        if self.deep_stem:
            layers = [self.stem]
        else:
            layers = [self.conv1, self.bn1, self.relu]
        layers.extend([self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4])
        self.features = nn.Sequential(*layers)

        if arch in ['resnet18', 'resnet34']:
            if self.deep_stem:
                self.nx_infos = {"2x": (1, 64), "4x": (3, 64), "8x": (4, 128), "16x": (5, 256), "32x": (6, 512)}
            else:
                self.nx_infos = {"2x": (3, 64), "4x": (5, 64), "8x": (6, 128), "16x": (7, 256), "32x": (8, 512)}
        else:
            if self.deep_stem:
                self.nx_infos = {"2x": (1, 64), "4x": (3, 256), "8x": (4, 512), "16x": (5, 1024), "32x": (6, 2048)}
            else:
                self.nx_infos = {"2x": (3, 64), "4x": (5, 256), "8x": (6, 512), "16x": (7, 1024), "32x": (8, 2048)}

    def _make_layer(self, block, planes, blocks, stride=1, with_dcn=False):
        """Build ResBlocks within one resnet stage.

        Args:
            block (nn.Module): ResBlock, one of ['BasicBlock', 'Bottleneck']
            planes (int): middle channel number in ResBlock.
            blocks (int): How many ResBlocks in one resnet stage.
            stride (int, optional): Stride for current resnet stage. Defaults to 1.
            with_dcn (bool, optional): DCN switch. Defaults to False.

        Returns:
            nn.Module: All ResBlocks within one resnet stage.
        """
        norm_layer = self._norm_layer
        downsample = None
        if self.avg_down and stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True),
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
            )

        elif stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer=norm_layer, with_dcn=with_dcn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                norm_layer=norm_layer, with_dcn=with_dcn))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(block, layers, pretrain, progress, **kwargs):
    """Entry function."""
    model = ResNet(block, layers, **kwargs)
    if pretrain:
        if kwargs.get('deep_stem', False):
            print(f'Failed to find cooresponding pretrain model for backbone config:\n{kwargs}')
        else:
            state_dict = load_state_dict_from_url(model_urls[kwargs['arch']], progress=progress)
            model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrain=False, progress=True, **kwargs):
    r"""ResNet-18 model from: Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnet18'
    return _resnet(BasicBlock, [2, 2, 2, 2], pretrain, progress, **kwargs)


def resnet34(pretrain=False, progress=True, **kwargs):
    r"""ResNet-34 model from: Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnet34'
    return _resnet(BasicBlock, [3, 4, 6, 3], pretrain, progress, **kwargs)


def resnet50(pretrain=False, progress=True, **kwargs):
    r"""ResNet-50 model from: Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnet50'
    return _resnet(Bottleneck, [3, 4, 6, 3], pretrain, progress, **kwargs)


def resnet101(pretrain=False, progress=True, **kwargs):
    r"""ResNet-101 model from: Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnet101'
    return _resnet(Bottleneck, [3, 4, 23, 3], pretrain, progress, **kwargs)


def resnet152(pretrain=False, progress=True, **kwargs):
    r"""ResNet-152 model from: Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnet152'
    return _resnet(Bottleneck, [3, 8, 36, 3], pretrain, progress, **kwargs)


def resnext50_32x4d(pretrain=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from.

    Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/pdf/1611.05431.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnext50_32x4d'
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], pretrain, progress, **kwargs)


def resnext101_32x8d(pretrain=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from.

    Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['arch'] = 'resnext101_32x8d'
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], pretrain, progress, **kwargs)


def make_backbone(name, pretrain=False, **kwargs):
    """Make backbone with registried name.

    Args:
        name (str): backbone name
        pretrain (bool): If True, loading a ImageNet or Places365 pretrained weight.
                         Default: False

    Returns:
        instance: backbone instance
    """
    available_models = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50': resnext50_32x4d,
        'resnext101': resnext101_32x8d,
    }
    assert name in available_models, f'Unsupported backbone: {name}'
    return available_models.get(name)(pretrain=pretrain, **kwargs)


if __name__ == '__main__':
    x = torch.randn(6, 3, 640, 640)
    kwargs = {'stage_with_dcn': [False, True, True, True], 'deep_stem': False, 'avg_down': False}
    name = 'resnet50'
    model = make_backbone(name, pretrain=True, **kwargs)
    print(f'model.nx_infos = {model.nx_infos }')
    y = model(x)
    print(y.shape)