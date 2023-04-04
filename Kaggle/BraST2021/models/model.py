import torch.nn as nn
import torch 
import torch.nn.functional as F

from monai.networks.blocks import MemoryEfficientSwish
from typing import Union, List, Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, conv_class, dropout_class, drop_p=0.0) -> None:
        super(ConvBlock, self).__init__()

        self.conv_layer = conv_class(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = nn.LeakyReLU()
        
        if drop_p > 0:
            self.drop = dropout_class(p=drop_p)
        else:
            self.drop = None

        self._init_layer_weights()
        
    def _init_layer_weights(self):
        for module in self.modules():
            if hasattr(module, 'weights'):
                # nn.init.xavier_uniform_(module.weight,)
                module.weight.data.normal_(mean=0.0, std=0.5)

    def forward(self, x):

        x = self.conv_layer(x)
        x = self.norm(x)
        
        if self.drop:
            x = self.drop(x)
        
        x = self.act(x)

        return x
    
class UNet_encoder(nn.Module):
    def __init__(self, in_dim, hidden_dims:Union[List, Tuple], spatial_dim, drop_p=0.0) -> None:
        super(UNet_encoder, self).__init__()

        if spatial_dim == 3:
            conv_class = nn.Conv3d
            dropout_class = nn.Dropout3d
            pooling_class = nn.MaxPool3d
        else:
            conv_class = nn.Conv2d
            dropout_class = nn.Dropout2d
            pooling_class = nn.MaxPool2d

        self.pool = pooling_class(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            ConvBlock(in_dim=in_dim        , out_dim=hidden_dims[0], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[0], out_dim=hidden_dims[0], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.layer2 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[0], out_dim=hidden_dims[1], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[1], out_dim=hidden_dims[1], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.layer3 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[1], out_dim=hidden_dims[2], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[2], out_dim=hidden_dims[2], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.layer4 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[2], out_dim=hidden_dims[3], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[3], out_dim=hidden_dims[3], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.layer5 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[3], out_dim=hidden_dims[4], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[4], out_dim=hidden_dims[4], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.layer6 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[4], out_dim=hidden_dims[5], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[5], out_dim=hidden_dims[5], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        

    def forward(self, x):

        stage_outputs = {}

        x = self.layer1(x)
        stage_outputs['stage1'] = x

        x = self.pool(x)
        x = self.layer2(x)
        stage_outputs['stage2'] = x

        x = self.pool(x)
        x = self.layer3(x)
        stage_outputs['stage3'] = x

        x = self.pool(x)
        x = self.layer4(x)
        stage_outputs['stage4'] = x

        x = self.pool(x)
        x = self.layer5(x)
        stage_outputs['stage5'] = x

        x = self.pool(x)
        x = self.layer6(x)

        return x, stage_outputs
    
class UNet_decoder(nn.Module):
    def __init__(self, out_dim, hidden_dims:Union[List, Tuple], spatial_dim, drop_p=0.0) -> None:
        super(UNet_decoder, self).__init__()

        if spatial_dim == 3:
            conv_class = nn.Conv3d
            dropout_class = nn.Dropout3d
            upconv_class = nn.ConvTranspose3d
        else:
            conv_class = nn.Conv2d
            dropout_class = nn.Dropout2d
            upconv_class = nn.ConvTranspose2d

        self.upconv0 = upconv_class(in_channels=hidden_dims[5], out_channels=hidden_dims[4], kernel_size=2, stride=2)
        self.layer0 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[4]*2, out_dim=hidden_dims[4], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[4], out_dim=hidden_dims[4], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.upconv1 = upconv_class(in_channels=hidden_dims[4], out_channels=hidden_dims[3], kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[3]*2, out_dim=hidden_dims[3], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[3], out_dim=hidden_dims[3], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.upconv2 = upconv_class(in_channels=hidden_dims[3], out_channels=hidden_dims[2], kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[2]*2, out_dim=hidden_dims[2], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[2], out_dim=hidden_dims[2], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.upconv3 = upconv_class(in_channels=hidden_dims[2], out_channels=hidden_dims[1], kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[1]*2, out_dim=hidden_dims[1], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[1], out_dim=hidden_dims[1], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.upconv4 = upconv_class(in_channels=hidden_dims[1], out_channels=hidden_dims[0], kernel_size=2, stride=2)
        self.layer4 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[0]*2, out_dim=hidden_dims[0], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[0], out_dim=hidden_dims[0], conv_class=conv_class, dropout_class=dropout_class, drop_p=drop_p)
        )

        self.fc = conv_class(in_channels=hidden_dims[0], out_channels=out_dim, kernel_size=1, stride=1)

        

    def forward(self, h, stage_outputs):

        h = self.upconv0(h)
        h = torch.concat([h, stage_outputs['stage5']], dim=1) 
        h = self.layer0(h)

        h = self.upconv1(h)
        h = torch.concat([h, stage_outputs['stage4']], dim=1) 
        h = self.layer1(h)

        h = self.upconv2(h)
        h = torch.concat([h, stage_outputs['stage3']], dim=1) 
        h = self.layer2(h)

        h = self.upconv3(h)
        h = torch.concat([h, stage_outputs['stage2']], dim=1) 
        h = self.layer3(h)

        h = self.upconv4(h)
        h = torch.concat([h, stage_outputs['stage1']], dim=1) 
        h = self.layer4(h)

        h = self.fc(h)

        return h
    

class UNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims:Union[Tuple, List], spatial_dim, dropout_p=0.0) -> None:
        super(UNet, self).__init__()
        assert spatial_dim in [2,3] and hidden_dims

        self.encoder = UNet_encoder(in_dim=input_dim, hidden_dims=hidden_dims, spatial_dim=spatial_dim, drop_p=dropout_p)
        self.decoder = UNet_decoder(out_dim=out_dim, hidden_dims=hidden_dims, spatial_dim=spatial_dim, drop_p=dropout_p)

    
    def forward(self, x):

        enc_out, stage_outputs = self.encoder(x)
        out = self.decoder(enc_out, stage_outputs)

        return out

