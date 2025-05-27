# Network
# 定义相关模块
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer


def get_conv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, groups: int = 1, kernel_size: int = 3, stride: int = 1,
        bias: bool = False, dilation=1):
    return Convolution(
        spatial_dims, in_channels, out_channels, strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True,
        groups=groups, dilation=dilation
    )


def get_upsample_layer(
        spatial_dims: int, in_channels: int, upsample_mode: UpsampleMode | str = "nontrainable", scale_factor: int = 2
):
    return UpSample(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )


class GMBI(nn.Module):
    # Group
    def __init__(self, in_channel, ksize=3):
        super().__init__()

        self.selected = int(in_channel // 4)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=self.selected, out_channels=self.selected, kernel_size=ksize, stride=1,
                      padding=1, groups=self.selected, dilation=1),
            nn.BatchNorm2d(self.selected),
            nn.ReLU(inplace=True),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.selected, out_channels=self.selected, kernel_size=ksize, stride=1,
                      padding=2, groups=self.selected, dilation=2),
            nn.BatchNorm2d(self.selected),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.selected, out_channels=self.selected, kernel_size=ksize, stride=1,
                      padding=3, groups=self.selected, dilation=3),
            nn.BatchNorm2d(self.selected),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.selected, out_channels=self.selected, kernel_size=ksize, stride=1,
                      padding=4, groups=self.selected, dilation=4),
            nn.BatchNorm2d(self.selected),
            nn.ReLU(inplace=True),

        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        identity = x
        # forward  promoting
        x0, x1, x2, x3 = torch.split(x, self.selected, dim=1)
        x0 = self.conv0(x0)
        x1 = self.conv1(self.sigmoid(x0) * x1 + x1)
        x2 = self.conv2(self.sigmoid(x1) * x2 + x2)
        x3 = self.conv3(self.sigmoid(x2) * x3 + x3)

        # Backword attention
        x3_score = self.sigmoid(x3)
        x2_score = self.sigmoid(x2)
        x1_score = self.sigmoid(x1)
        x0_p = x1_score * x0 + x0
        x1_p = x2_score * x1 + x1
        x2_p = x3_score * x2 + x2

        fusion = torch.cat([x0_p, x1_p, x2_p, x3], dim=1)

        out = self.fusion(fusion) + identity
        # out = channel_shuffle(x=out, groups=4, spatial_dims=2)
        return out


def channel_shuffle(x, groups, spatial_dims=3):
    if spatial_dims == 2:
        b, c, h, w = x.size()  # 获取输入特征图的shape=[b,c,h,w]
        c_g = c // groups  # 均分通道，获得每个组对应的通道数
        x = x.view(b, groups, c_g, h, w)  # 特征图shape调整 [b,c,h,w]==>[b,g,c_g,h,w]
        # 维度调整 [b,g,c_g,h,w]==>[b,c_g,g,h,w]；将调整后的tensor以连续值的形式保存在内存中
        x = torch.transpose(x, 1, 2).contiguous()  # 将调整后的通道拼接回去 [b,c_g,g,h,w]==>[b,c,h,w]
        x = x.view(b, -1, h, w)  # 完成通道重排
    else:
        b, c, d, h, w = x.size()
        c_g = c // groups  # 均分通道，获得每个组对应的通道数
        x = x.view(b, groups, c_g, d, h, w)  # 特征图shape调整 [b,c,d, h,w]==>[b,g,c_g,d, h,w]
        # 维度调整 [b,g,c_g,d, h,w]==>[b,c_g,g,d, h,w]；将调整后的tensor以连续值的形式保存在内存中
        x = torch.transpose(x, 1, 2).contiguous()  # 将调整后的通道拼接回去 [b,c_g,g,d, h,w]==>[b,c,d, h,w]
        x = x.view(b, -1, d, h, w)  # 完成通道重排

    return x


class DownBlock(nn.Module):
    # down-sample operation
    def __init__(self, in_channel, out_channel, *args):
        super().__init__(*args)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=2, padding=1,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, *args):
        super().__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(x.size())
        out = self.conv(x)
        return out


class DecoderFusion(nn.Module):
    def __init__(self, in_channel, out_channel, ):
        super(DecoderFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

        )

    def forward(self, low, high):
        out = self.conv(low + high)
        return out


class GMBINet(nn.Module):
    def __init__(self, in_ch: int = 3, init_filters: int = 16, out_channel: int = 1,
                 blocks_down=(1, 3, 4, 6, 3), ksize=3):
        super().__init__()

        ch_list = (init_filters * 1, init_filters * 2, init_filters * 4, init_filters * 6, init_filters * 8)
        # stage 1 16
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=ch_list[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU(inplace=True),
        )
        # stage 2 32
        self.encoder2 = nn.Sequential(
            DownBlock(in_channel=ch_list[0], out_channel=ch_list[1]),
            *[GMBI(in_channel=ch_list[1], ksize=ksize) for _ in range(blocks_down[1])]
        )

        # stage 3 64
        self.encoder3 = nn.Sequential(
            DownBlock(in_channel=ch_list[1], out_channel=ch_list[2]),
            *[GMBI(in_channel=ch_list[2], ksize=ksize) for _ in range(blocks_down[2])]
        )
        # stage 4 128
        self.encoder4 = nn.Sequential(
            DownBlock(in_channel=ch_list[2], out_channel=ch_list[3]),
            *[GMBI(in_channel=ch_list[3], ksize=ksize) for _ in range(blocks_down[3])]
        )
        # stage 5
        self.encoder5 = nn.Sequential(
            DownBlock(in_channel=ch_list[3], out_channel=ch_list[4]),
            *[GMBI(in_channel=ch_list[4], ksize=ksize) for _ in range(blocks_down[4])]
        )

        # Decoder
        self.decoder5 = ConvBlock(in_channel=ch_list[4], out_channel=ch_list[3])
        self.decoder4 = ConvBlock(in_channel=ch_list[3], out_channel=ch_list[2])
        self.decoder3 = ConvBlock(in_channel=ch_list[2], out_channel=ch_list[1])
        self.decoder2 = ConvBlock(in_channel=ch_list[1], out_channel=ch_list[0])
        self.decoder1 = ConvBlock(in_channel=ch_list[0], out_channel=ch_list[0])

        # skip connection

        self.DF4 = DecoderFusion(in_channel=ch_list[3], out_channel=ch_list[2])
        self.DF3 = DecoderFusion(in_channel=ch_list[2], out_channel=ch_list[1])
        self.DF2 = DecoderFusion(in_channel=ch_list[1], out_channel=ch_list[0])
        self.DF1 = DecoderFusion(in_channel=ch_list[0], out_channel=ch_list[0])
        # super version

        self.conv_out1 = nn.Sequential(

            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=init_filters, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_out2 = nn.Sequential(

            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=ch_list[0], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conv_out3 = nn.Sequential(

            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=ch_list[1], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_out4 = nn.Sequential(

            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=ch_list[2], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_out5 = nn.Sequential(

            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=ch_list[3], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_final = nn.Sequential(

            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=out_channel * 5, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder for feature extraction
        encoder1 = self.encoder1(x)  # 1/2
        encoder2 = self.encoder2(encoder1)  # 1/ 4
        encoder3 = self.encoder3(encoder2)  # 1/8
        encoder4 = self.encoder4(encoder3)  # 1/16
        encoder5 = self.encoder5(encoder4)  # 1/32

        # Decoder process
        # Decoder4
        encoder5_up = self.decoder5(encoder5)
        out5 = self.conv_out5(encoder5_up)  # 用于生成监督
        up4 = F.interpolate(encoder5_up, encoder4.size()[2:])  # Decoder5

        encoder4_up = self.DF4(encoder4, up4)
        out4 = self.conv_out4(encoder4_up)
        up3 = F.interpolate(encoder4_up, encoder3.size()[2:])  # Decoder4

        encoder3_up = self.DF3(encoder3, up3)
        out3 = self.conv_out3(encoder3_up)
        up2 = F.interpolate(encoder3_up, encoder2.size()[2:])  # Decoder3

        encoder2_up = self.DF2(encoder2, up2)
        out2 = self.conv_out2(encoder2_up)
        up1 = F.interpolate(encoder2_up, encoder1.size()[2:])  # Decoder2

        encoder1_up = self.DF1(encoder1, up1)  # Decoder1
        out1 = self.conv_out1(encoder1_up)
        out5 = F.interpolate(out5, x.size()[2:])
        out4 = F.interpolate(out4, x.size()[2:])
        out3 = F.interpolate(out3, x.size()[2:])
        out2 = F.interpolate(out2, x.size()[2:])
        out1 = F.interpolate(out1, x.size()[2:])

        out = [out1, out2, out3, out4, out5]

        return out


def get_intermediate_output(model, input_data, layer_name):
    for name, layer in model.named_modules():
        if name == layer_name:
            out = layer(input_data)
            return out


if __name__ == "__main__":
    model = GMBINet(init_filters=16, blocks_down=(1, 3, 4, 6, 3), out_channel=1, ksize=3)

    batch_size = 1
    x = torch.randn([batch_size, 3, 2048, 2048])

    # Calculate FLOPs and Params
    from thop import profile

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
