import torch
import torch.nn as nn

from typing import Union, List, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, spatial_dim, drop_p=0.0) -> None:
        super(ConvBlock, self).__init__()

        if spatial_dim == 3:
            self.conv = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm = nn.InstanceNorm3d(num_features=out_dim, affine=True)
            self.drop = nn.Dropout3d(p=drop_p) if drop_p else nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm = nn.InstanceNorm2d(num_features=out_dim, affine=True)
            self.drop = nn.Dropout2d(p=drop_p) if drop_p else nn.Identity()

        self.act = nn.LeakyReLU()

        self._init_layer_weights()
        
    def _init_layer_weights(self):
        for module in self.modules():
            if hasattr(module, 'weights'):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')  # relu, leaky_relu, selu

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.act(x)

        return x
    

class encoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, spatial_dim, drop_p) -> None:
        super(encoder, self).__init__()

        self.pool = nn.MaxPool3d(2,2) if spatial_dim == 3 else nn.MaxPool2d(2,2)

        self.layer1 = nn.Sequential(
                            ConvBlock(in_dim=in_dim,         out_dim=hidden_dims[0], spatial_dim=spatial_dim, drop_p=drop_p),
                            ConvBlock(in_dim=hidden_dims[0], out_dim=hidden_dims[0], spatial_dim=spatial_dim, drop_p=drop_p),
                        )
        
        self.layer2 = nn.Sequential(
                            ConvBlock(in_dim=hidden_dims[0], out_dim=hidden_dims[1], spatial_dim=spatial_dim, drop_p=drop_p),
                            ConvBlock(in_dim=hidden_dims[1], out_dim=hidden_dims[1], spatial_dim=spatial_dim, drop_p=drop_p),
                        )
        
        self.layer3 = nn.Sequential(
                            ConvBlock(in_dim=hidden_dims[1], out_dim=hidden_dims[2], spatial_dim=spatial_dim, drop_p=drop_p),
                            ConvBlock(in_dim=hidden_dims[2], out_dim=hidden_dims[2], spatial_dim=spatial_dim, drop_p=drop_p),
                        )
        
        self.layer4 = nn.Sequential(
                            ConvBlock(in_dim=hidden_dims[2],  out_dim=hidden_dims[3], spatial_dim=spatial_dim, drop_p=drop_p),
                            ConvBlock(in_dim=hidden_dims[3], out_dim=hidden_dims[3], spatial_dim=spatial_dim, drop_p=drop_p),
                        )
        
        self.layer5 = nn.Sequential(
                            ConvBlock(in_dim=hidden_dims[3],  out_dim=hidden_dims[4], spatial_dim=spatial_dim, drop_p=drop_p),
                            ConvBlock(in_dim=hidden_dims[4], out_dim=hidden_dims[4], spatial_dim=spatial_dim, drop_p=drop_p),
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

        return x, stage_outputs
    
class decoder(nn.Module):
    def __init__(self, hidden_dims, out_dim, spatial_dim, drop_p) -> None:
        super(decoder, self).__init__()

        upconv_class = nn.ConvTranspose3d if spatial_dim == 3 else nn.ConvTranspose2d
        projection = nn.Conv3d if spatial_dim == 3 else nn.Conv2d

        self.upconv1 = upconv_class(in_channels=hidden_dims[4], out_channels=hidden_dims[3], kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[3]*2, out_dim=hidden_dims[3], spatial_dim=spatial_dim, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[3], out_dim=hidden_dims[3], spatial_dim=spatial_dim, drop_p=drop_p)
        )

        self.upconv2 = upconv_class(in_channels=hidden_dims[3], out_channels=hidden_dims[2], kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[2]*2, out_dim=hidden_dims[2], spatial_dim=spatial_dim, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[2], out_dim=hidden_dims[2], spatial_dim=spatial_dim, drop_p=drop_p)
        )

        self.upconv3 = upconv_class(in_channels=hidden_dims[2], out_channels=hidden_dims[1], kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[1]*2, out_dim=hidden_dims[1], spatial_dim=spatial_dim, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[1], out_dim=hidden_dims[1], spatial_dim=spatial_dim, drop_p=drop_p)
        )

        self.upconv4 = upconv_class(in_channels=hidden_dims[1], out_channels=hidden_dims[0], kernel_size=2, stride=2)
        self.layer4 = nn.Sequential(
            ConvBlock(in_dim=hidden_dims[0]*2, out_dim=hidden_dims[0], spatial_dim=spatial_dim, drop_p=drop_p),
            ConvBlock(in_dim=hidden_dims[0], out_dim=hidden_dims[0], spatial_dim=spatial_dim, drop_p=drop_p)
        )

        # self.fc = projection(in_channels=hidden_dims[0], out_channels=out_dim, kernel_size=1, stride=1)
    
    def forward(self, h, stage_outputs):
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

        # h = self.fc(h)

        return h
    

class UNet_tailF(nn.Module):
    def __init__(self, spatial_dim, input_dim, out_dim, hidden_dims:Union[Tuple, List], dropout_p=0.0,
                 use_MS=False, MS_pt='/root/Competitions/MICCAI/AutoPET2023/example/Genesis_Chest_CT.pt') -> None:
        super(UNet_tailF, self).__init__()
        assert spatial_dim in [2,3] and hidden_dims
        self.use_MS = use_MS
        input_dim = input_dim//2
        # hidden_dims = [dim//2 for dim in hidden_dims]

        if use_MS:
            from model.model_genesis_UNet import UNet3D
            from collections import OrderedDict

            loaded_state_dict = torch.load(MS_pt)['state_dict']
            new_state_dict = OrderedDict()
            for n, v in loaded_state_dict.items():
                name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
                new_state_dict[name] = v

            self.CT_net = UNet3D()
            self.CT_net.load_state_dict(new_state_dict) 
            
        else:
            self.CT_encoder = encoder(in_dim=input_dim, hidden_dims=hidden_dims, spatial_dim=spatial_dim, drop_p=dropout_p)
            self.CT_decoder = decoder(out_dim=out_dim, hidden_dims=hidden_dims, spatial_dim=spatial_dim, drop_p=dropout_p)

        self.PET_encoder = encoder(in_dim=input_dim, hidden_dims=hidden_dims, spatial_dim=spatial_dim, drop_p=dropout_p)
        self.PET_decoder = decoder(out_dim=out_dim, hidden_dims=hidden_dims, spatial_dim=spatial_dim, drop_p=dropout_p)

        self.fc = nn.Conv3d(in_channels=hidden_dims[0]*2, out_channels=out_dim, kernel_size=1, stride=1)
        
    def forward(self, x):
        ct, pet = torch.chunk(x,chunks=2, dim=1)

        if self.use_MS:
            CT_out = self.CT_net(ct) # output dim = 64
        else: 
            enc_out, stage_outputs = self.CT_encoder(ct)
            CT_out = self.CT_decoder(enc_out, stage_outputs)

        enc_out, stage_outputs = self.PET_encoder(pet)
        PET_out = self.PET_decoder(enc_out, stage_outputs)

        # print(f'{PET_out.shape=}')
        # print(f'{CT_out.shape=}')
        out = torch.concat([CT_out, PET_out], dim=1)
        # print(f'{out.shape=}')

        out = self.fc(out)

        return out
