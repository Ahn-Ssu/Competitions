# Monai UNet based model
# returning all intermediate variables

from typing import Sequence, Union, Tuple, Optional

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm


class MyUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError(
                "the length of `strides` should equal to `len(channels) - 1`."
            )
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError(
                    "the length of `kernel_size` should equal to `dimensions`."
                )
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError(
                    "the length of `up_kernel_size` should equal to `dimensions`."
                )

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        

        inc = [in_channels] + list(channels[:-1])  # C
        upc = [channels[i] * 2 for i in range(len(channels) - 2)] + [
            channels[-2] + channels[-1]
        ]
        outc = [out_channels] + list(channels[:-2])  # C-1

        self.down_blocks = nn.ModuleList(
            [
                self._get_down_layer(inc[i], inc[i + 1], strides[i], i == 0)
                for i in range(len(inc) - 1)
            ]
        )
        self.bottom_block = self._get_bottom_layer(channels[-2], channels[-1])
        self.up_blocks = nn.ModuleList(
            [
                self._get_up_layer(upc[i], outc[i], strides[i], i == 0)
                for i in range(len(outc))
            ]
        )
        self.softmax = nn.Softmax(dim=1)


    def forward_down(self, x):
        output = []
        for block in self.down_blocks:
            x = block(x)
            output.append(x)
        return output

    def forward_bottom(self, x):
        return self.bottom_block(x)

    def forward_up(self, down_output, bottom):
        x = bottom
        output = []
        for i in range(len(self.up_blocks) - 1, -1, -1):
            z = torch.cat([down_output[i], x], dim=1)
            x = self.up_blocks[i](z)
            output.append(x)
        return output

    def forward(self, x):

        down = self.forward_down(x)
        bottom = self.forward_bottom(down[-1])
        up = self.forward_up(down, bottom)
        output = up[-1]#self.softmax(up[-1])
        return down, bottom, up, output

    def _get_down_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool
    ) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool
    ) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv