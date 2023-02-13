import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torchvision.models as models


def _make_divisible(ch, diviisor=8, min_ch=None):
    if min_ch is None:
        min_ch = diviisor

    new_ch = max(min_ch, int(ch + diviisor / 2) // diviisor * diviisor)
    if new_ch < 0.9 * ch:
        new_ch += diviisor
    return new_ch


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    def __init__(
        self,
        kernel: int,
        input_c: int,
        out_c: int,
        expanded_ratio: int,  # 1 or 6
        stride: int,  # 1 or 2
        use_se: bool,  # True
        drop_rate: float,
        index: str,  # 1a, 2a, 2b, ...
        width_coefficient: float,
    ):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(
        self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]
    ):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = cnf.stride == 1 and cnf.input_c == cnf.out_c
        layers = OrderedDict()
        activation_layer = nn.SiLU

        if cnf.expanded_c != cnf.input_c:
            layers.update(
                (
                    {
                        "expand_conv": ConvBNActivation(
                            cnf.input_c,
                            cnf.expanded_c,
                            kernel_size=1,
                            norm_layer=norm_layer,
                            activation_layer=activation_layer,
                        )
                    }
                )
            )

        layers.update(
            {
                "dwconv": ConvBNActivation(
                    cnf.expanded_c,
                    cnf.expanded_c,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    groups=cnf.expanded_c,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            }
        )
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c, cnf.expanded_c)})
        layers.update(
            {
                "project_conv": ConvBNActivation(
                    cnf.expanded_c,
                    cnf.out_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                )
            }
        )
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        reslut = self.block(x)
        reslut = self.dropout(reslut)
        if self.use_res_connect:
            reslut += x
        return reslut


class EfficientNet(nn.Module):
    def __init__(
        self,
        width_coefficient: float,
        depth_coefficient: float,
        num_classes: int = 1000,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(EfficientNet, self).__init__()

        default_cnf = [
            [3, 32, 16, 1, 1, True, drop_connect_rate, 1],
            [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
            [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
            [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
            [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
            [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
            [3, 192, 320, 6, 1, True, drop_connect_rate, 1],
        ]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(
            InvertedResidualConfig.adjust_channels, width_coefficient=width_coefficient
        )

        bneck_conf = partial(
            InvertedResidualConfig, width_coefficient=width_coefficient
        )

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        layers = OrderedDict()

        layers.update(
            {
                "stem_conv": ConvBNActivation(
                    in_planes=3,
                    out_planes=adjust_channels(32),
                    kernel_size=3,
                    stride=2,
                    norm_layer=norm_layer,
                )
            }
        )
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update(
            {
                "top": ConvBNActivation(
                    in_planes=last_conv_input_c,
                    out_planes=last_conv_output_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                )
            }
        )

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=0.2,
        num_classes=num_classes,
    )


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        dropout_rate=0.2,
        num_classes=num_classes,
    )


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        dropout_rate=0.3,
        num_classes=num_classes,
    )


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        dropout_rate=0.3,
        num_classes=num_classes,
    )


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(
        width_coefficient=1.4,
        depth_coefficient=1.8,
        dropout_rate=0.4,
        num_classes=num_classes,
    )


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(
        width_coefficient=1.6,
        depth_coefficient=2.2,
        dropout_rate=0.4,
        num_classes=num_classes,
    )


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(
        width_coefficient=1.8,
        depth_coefficient=2.6,
        dropout_rate=0.5,
        num_classes=num_classes,
    )


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(
        width_coefficient=2.0,
        depth_coefficient=3.1,
        dropout_rate=0.5,
        num_classes=num_classes,
    )


models_1 = models.resnext50_32x4d(pretrained=False)