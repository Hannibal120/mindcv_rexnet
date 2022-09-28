from math import ceil

import mindspore.nn as nn

from .layers import Conv2dNormActivation, SqueezeExcite
from .registry import register_model
from .utils import load_pretrained, make_divisible

__all__ = [
    'rexnet_x09',
    'rexnet_x10',
    'rexnet_x13',
    'rexnet_x15',
    'rexnet_x20'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'rexnet_x09': _cfg(url=''),
    'rexnet_x10': _cfg(url=''),
    'rexnet_x13': _cfg(url=''),
    'rexnet_x15': _cfg(url=''),
    'rexnet_x20': _cfg(url='')
}


class LinearBottleneck(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_ratio,
                 stride,
                 use_se=True,
                 se_ratio=1/12,
                 ch_div=1,
                 act_layer=nn.SiLU,
                 dw_act_layer=nn.ReLU6,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        if exp_ratio != 1:
            dw_channels = in_channels * exp_ratio
            self.conv_exp = Conv2dNormActivation(in_channels, dw_channels, 1, activation=act_layer)
        else:
            dw_channels = in_channels
            self.conv_exp = None

        self.conv_dw = Conv2dNormActivation(dw_channels, dw_channels, 3, stride, padding=1,
                                        groups=dw_channels, activation=None)

        if use_se:
            self.se = SqueezeExcite(dw_channels, rd_channels=make_divisible(int(dw_channels * se_ratio), ch_div),
                                     norm=nn.BatchNorm2d)
        else:
            self.se = None
        self.act_dw = nn.ReLU6()

        self.conv_pwl = Conv2dNormActivation(dw_channels, out_channels, 1, padding=0, activation=None)

    def construct(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            x[:, 0:self.in_channels] += shortcut
        return x


def _build_block():
    pass


class ReXNetV1(nn.Cell):
    def __init__(self, 
                 in_channels=3, 
                 fi_channels=180,
                 initial_channels = 16,
                 width_mult=1.0, 
                 depth_mult=1.0, 
                 num_classes=1000,
                 use_se=True,
                 se_ratio=1/12,
                 dropout_ratio=0.2,
                 ch_div=1,
                 act_layer=nn.SiLU,
                 dw_act_layer=nn.ReLU6,
                 cls_useconv=False):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = initial_channels / width_mult if width_mult < 1.0 else initial_channels

        features = []
        in_channels_group = []
        out_channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                out_channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += fi_channels / (self.depth // 3 * 1.0)
                out_channels_group.append(int(round(inplanes * width_mult)))

        stem_chs = make_divisible(round(stem_channel * width_mult), divisor=ch_div)
        self.stem = Conv2dNormActivation(in_channels, stem_chs, stride=2, padding=1, activation=act_layer)

        for _, (in_c, out_c, exp_ratio, stride, use_se) in enumerate(zip(in_channels_group, 
                                                                         out_channels_group, 
                                                                         exp_ratios, 
                                                                         strides, 
                                                                         use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             out_channels=out_c,
                                             exp_ratio=exp_ratio,
                                             stride=stride,
                                             use_se=use_se, 
                                             se_ratio=se_ratio,
                                             act_layer=act_layer,
                                             dw_act_layer=dw_act_layer))

        pen_channels = make_divisible(int(1280 * width_mult), divisor=ch_div)
        features.append(Conv2dNormActivation(out_channels_group[-1], 
                                             pen_channels, 
                                             kernel_size=1, 
                                             activation=act_layer))

        features.append(nn.AdaptiveAvgPool2d(1))
        self.useconv = cls_useconv
        self.features = nn.SequentialCell(*features)
        if self.useconv:
            self.cls = nn.SequentialCell(
                nn.Dropout(dropout_ratio),
                nn.Conv2d(pen_channels, num_classes, 1, has_bias=True))
        else:
            self.cls = nn.SequentialCell(
                nn.Dropout(dropout_ratio),
                nn.Dense(pen_channels, num_classes))

    def extract_features(self, x):
        return self.cls[:-1](x)
    
    def construct(self, x):
        x = self.stem(x)
        x = self.features(x)
        if not self.useconv:
            x = x.reshape((-1, x.shape[1]))
            x = self.cls(x)
        else:
            x = self.cls(x).reshape((-1, x.shape[1]))
        return x


@register_model
def rexnet_x09(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    default_cfg = default_cfgs['rexnet_x09']
    model = ReXNetV1(width_mult=0.9, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def rexnet_x10(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    default_cfg = default_cfgs['rexnet_x10']
    model = ReXNetV1(width_mult=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def rexnet_x13(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    default_cfg = default_cfgs['rexnet_x13']
    model = ReXNetV1(width_mult=1.3, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def rexnet_x15(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    default_cfg = default_cfgs['rexnet_x15']
    model = ReXNetV1(width_mult=1.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def rexnet_x20(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ReXNetV1:
    default_cfg = default_cfgs['rexnet_x20']
    model = ReXNetV1(width_mult=2.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
