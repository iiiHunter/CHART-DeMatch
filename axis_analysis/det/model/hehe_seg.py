#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: work5_20201215
@Author: sol@JinGroup
@File: hehe_seg.py
@Time: 1/27/21 1:34 PM
@E-mail: hesuozhang@gmail.com
'''

"""
HeHe used model
"""
import math
import torch
import torch.nn.init as init
import torch.nn as nn

from det.model.hehe_resnet import make_backbone
from collections import OrderedDict
import torch.nn.functional as F


def kaiming_weight_init(m):
    """Init weights by kaiming.

    Args:
        m (nn.Module): input module
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.RNN):
        init.kaiming_normal_(m.weight_ih_l0.data)
        init.kaiming_normal_(m.weight_hh_l0.data)
        if m.bias_ih_l0 is not None:
            m.bias_ih_l0.data.zero_()
        if m.bias_hh_l0 is not None:
            m.bias_hh_l0.data.zero_()


class ConvBnRelu(nn.Module):
    """Convolution-BatchNormliztion-ReLU Sequential."""

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, dilation, groups, bias),
                  nn.BatchNorm2d(out_channels)]
        self.relu = nn.ReLU(inplace=True)

        self.m = nn.Sequential(*layers)
        for m in self.m.modules():
            kaiming_weight_init(m)

    def forward(self, x):
        return self.relu(self.m(x))


def make_submodule(idx_start, idx_end, modules):
    m = nn.Sequential()
    for i in range(idx_start, idx_end):
        m.add_module(str(i), modules[i])
    return m


class Paramount(nn.Module):
    """Paramount Class."""

    def __init__(self, backbone, top_nx=32, pretrain=False):
        """Paramount Class.

        Get Paramount feature maps

        Args:
            backbone (str): which backbone to build
            top_nx (int): the topset nx feature map, if top_nx = 32, Paramount
                    will return (2x, 4x, ..., 32x) feature maps after forweard
            pretrain (bool): whether to use pretrained weight for backbone
        """
        super().__init__()
        assert top_nx == 32
        self.backbone = backbone

        backbone = make_backbone(backbone, pretrain, stage_with_dcn=[False] * 4)
        features = backbone.features
        nx_infos = backbone.nx_infos
        idxs = [nx_infos[key][0] for key in ['2x', '4x', '8x', '16x', '32x']]
        channels = [nx_infos[key][1] for key in ['2x', '4x', '8x', '16x', '32x']]
        self.total_stage_num = 5

        layers = []
        self.nx_infos = OrderedDict()
        idxs.insert(0, 0)
        channels.insert(0, 0)
        for j in range(1, self.total_stage_num + 1):
            self.nx_infos[str(2 ** j) + 'x'] = (j - 1, channels[j])
            layers.append(make_submodule(idxs[j - 1], idxs[j], features))
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        features = []
        for i in range(self.total_stage_num):
            x = self.m[i](x)
            features.append(x)
        return features


class FPN(nn.Module):
    """Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf."""

    def __init__(self, backbone, pretrain=True, out_nx_infos={'1x': 32}):
        super().__init__()
        top_nx = 32
        self.out_nx_infos = out_nx_infos
        self.paramount = Paramount(backbone, top_nx=top_nx, pretrain=pretrain)
        self.paramount_nx_infos = self.paramount.nx_infos
        self.bridge_out_chs = {
            '2x': 64,
            '4x': 128,
            '8x': 128,
            '16x': 256,
            '32x': 256,
            '64x': 512,
            '128x': 512,
            '256x': 512,
        }

        self.nx_infos = OrderedDict()
        self.nx_min = int(math.log2(min(int(v[:-1]) for v in out_nx_infos.keys())))
        self.nx_max = int(math.log2(top_nx))

        nx_counter = 0
        self.bridge = nn.Sequential()
        self.conv3x3_after_fusion = nn.Sequential()
        self.output_conv1x1 = nn.Sequential()

        for nx in range(self.nx_min, self.nx_max + 1):
            key = str(2 ** nx) + 'x'
            in_ch = self.paramount_nx_infos[key][1]
            out_ch = self.bridge_out_chs[key]
            if key == "2x":
                self.bridge.add_module(key, self.build_bridge("1x1", in_ch=in_ch, out_ch=out_ch))
            else:
                self.bridge.add_module(key, self.build_bridge("3x3", in_ch=in_ch, out_ch=out_ch))

            if 2 ** nx < top_nx:
                pre_out_ch = self.bridge_out_chs[str(2 ** (nx + 1)) + 'x']
                self.conv3x3_after_fusion.add_module(key, self.build_bridge("3x3",
                                                                            in_ch=pre_out_ch + out_ch,
                                                                            out_ch=out_ch))

            if key in out_nx_infos:
                self.nx_infos[key] = (nx_counter, out_nx_infos[key])
                self.output_conv1x1.add_module(key, self.build_bridge("1x1", in_ch=out_ch, out_ch=out_nx_infos[key]))
                nx_counter += 1

    def paramount_forward(self, x):
        """Extract features from paramount."""
        feats = self.paramount(x)
        outs = {}
        # bug here
        for i, feat in enumerate(feats):
            outs["{}x".format(2 ** (i + 1))] = feat
        return outs

    def bridge_forward(self, in_feats):
        """Apply convolution or other module on paramount output features."""
        outs = {}
        for key, value in in_feats.items():
            nx = math.log2(int(key.split('x')[0]))
            if nx >= self.nx_min and nx <= self.nx_max:
                outs[key] = getattr(self.bridge, key)(value)
        return outs

    def build_bridge(self, block_type, in_ch, out_ch):
        """Build bridge."""
        if block_type == "1x1":
            return ConvBnRelu(in_ch, out_ch, kernel_size=1, bias=False)
        elif block_type == "3x3":
            return nn.Sequential(
                ConvBnRelu(in_ch, out_ch, 3, 1, 1, bias=False),
                ConvBnRelu(out_ch, out_ch, 3, 1, 1, bias=False)
            )
        else:
            raise ValueError

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): input images.

        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        features = self.paramount_forward(x)
        # comment feature fusion
        features = self.bridge_forward(features)

        top_feature = None
        outputs = []
        for nx in range(self.nx_min, self.nx_max + 1)[::-1]:
            key = str(2 ** nx) + 'x'
            feature = features[key]
            if top_feature is not None:
                assert top_feature.size(2) == feature.size(2) // 2
                assert top_feature.size(3) == feature.size(3) // 2
                up_feature = F.interpolate(top_feature, size=feature.size()[2:],
                                           mode='bilinear', align_corners=False)

                feature = torch.cat([up_feature, feature], dim=1)
                feature = getattr(self.conv3x3_after_fusion, key)(feature)
            top_feature = feature
            if key in self.nx_infos:
                outputs.append(getattr(self.output_conv1x1, key)(feature))
        return outputs[::-1]


class SegModel(nn.Module):
    """Tmp model for implementing key point detection."""

    def __init__(self, cfg):
        super().__init__()
        assert 'out_nx_infos' in cfg and len(cfg['out_nx_infos']) == 1
        assert list(cfg['out_nx_infos'].keys())[0] in ['1x', '2x']

        self.out_nx = '2x' if list(cfg['out_nx_infos'].keys())[0] == '2x' else '1x'
        self.backbone_out_ch = cfg['out_nx_infos'][self.out_nx]
        if self.out_nx == '1x':
            cfg['out_nx_infos'] = {'2x': self.backbone_out_ch}

        self.backbone = FPN(**cfg)
        self.head_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_ch, self.backbone_out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.backbone_out_ch, self.backbone_out_ch, kernel_size=1),
        )
        self.pred = nn.Conv2d(self.backbone_out_ch, 1, kernel_size=1)

    def forward(self, x):
        feature = self.backbone(x)[0]
        feature = self.head_conv(feature)
        out = self.pred(feature)
        if self.out_nx == '1x':
            out = F.interpolate(out, scale_factor=2, mode='bilinear')
        return out


def create_model():
    cfg = {'backbone': 'resnet18', 'out_nx_infos': {'1x': 32}, 'pretrain': True}
    model = SegModel(cfg)
    return model


if __name__ == '__main__':
    x = torch.randn(6, 3, 640, 640)
    cfg = {'backbone': 'resnet18', 'out_nx_infos': {'1x': 32}, 'pretrain': True}
    model = SegModel(cfg)
    print(model)
    y = model(x)
    print(y.shape)
