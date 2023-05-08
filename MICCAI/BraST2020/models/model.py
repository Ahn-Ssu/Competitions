import torch.nn as nn
import torch 
import torch.nn.functional as F

from monai.networks.blocks import MemoryEfficientSwish


class DeepSEED(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout_p=0.1) -> None:
        super(DeepSEED, self).__init__()

        self.encoder = Encoder(input_dim=in_dim, hidden_dim=hidden_dims, dropout_p=dropout_p)
        self.decoder = Decoder(hidden_dim=hidden_dims, out_dim=out_dim, dropout_p=dropout_p)

    def forward(self, x):

        enc_out, stage_outputs = self.encoder(x)
        out = self.decoder(enc_out, stage_outputs)

        return out
class SqueezeExcitation3D(nn.Module):
    def __init__(self, in_dim, reduction_ratio=4) -> None:
        super(SqueezeExcitation3D, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1) # 1x1x1xC

        self.excitation = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=in_dim//reduction_ratio, kernel_size=1, stride=1, bias=False),
            nn.SiLU(),
            nn.Conv3d(in_channels=in_dim//reduction_ratio, out_channels=in_dim, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        se_out = self.squeeze(x)

        se_out = self.excitation(se_out)

        return x * se_out



class FusedMBConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, expansion_ratio=4, squeeze_ratio=4, stride=1, dropout_p=0.1) -> None:
        super(FusedMBConv3d, self).__init__()
        self.use_residual = in_dim == out_dim and stride == 1 
        hidden_dim = int(in_dim * expansion_ratio)
        padding = 1

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.SiLU(),
            nn.Dropout3d(p=dropout_p)
        )

        self.se = SqueezeExcitation3D(in_dim=hidden_dim, reduction_ratio=squeeze_ratio)

        self.projection = nn.Sequential(
            nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(out_dim),
            nn.Dropout3d(p=dropout_p)
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = self.se(h)
        h = self.projection(h)

        if self.use_residual:
            h = h + x
        
        return h
    


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.1) -> None:
        super(Encoder, self).__init__()

        # layers = 
        self.conv1 = nn.Sequential(*[FusedMBConv3d(in_dim=input_dim, out_dim=hidden_dim[0], stride=2, dropout_p=dropout_p)]+ \
              [FusedMBConv3d(in_dim=hidden_dim[0], out_dim=hidden_dim[0], stride=1, dropout_p=dropout_p) for _ in range(1)])

        self.conv2 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[0], out_dim=hidden_dim[1], stride=2, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[1], out_dim=hidden_dim[1], stride=1, dropout_p=dropout_p) for _ in range(2)]
        )

        self.conv3 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[1], out_dim=hidden_dim[2], stride=2, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[2], out_dim=hidden_dim[2], stride=1, dropout_p=dropout_p) for _ in range(3)]
        )

        self.conv4 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[2], out_dim=hidden_dim[3], stride=2, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[3], out_dim=hidden_dim[3], stride=1, dropout_p=dropout_p) for _ in range(4)]
        )

    def forward(self, x):

        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        stage_outputs = [h1, h2, h3]

        return h4, stage_outputs
    

class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, dropout_p=0.1):
        super(Decoder, self).__init__()

        
        self.up_conv1 = nn.ConvTranspose3d(hidden_dim[3], hidden_dim[2], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[3], out_dim=hidden_dim[2], stride=1, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[2], out_dim=hidden_dim[2], stride=1, dropout_p=dropout_p) for _ in range(4) ]
        )

        self.up_conv2 = nn.ConvTranspose3d(hidden_dim[2], hidden_dim[1], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[2], out_dim=hidden_dim[1], stride=1, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[1], out_dim=hidden_dim[1], stride=1, dropout_p=dropout_p) for _ in range(3) ]
        )

        self.up_conv3 = nn.ConvTranspose3d(hidden_dim[1], hidden_dim[0], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[1], out_dim=hidden_dim[0], stride=1, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[0], out_dim=hidden_dim[0], stride=1, dropout_p=dropout_p) for _ in range(2) ]
        )

        self.up_conv4 = nn.ConvTranspose3d(hidden_dim[0], hidden_dim[0], kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(*[FusedMBConv3d(in_dim=hidden_dim[0], out_dim=hidden_dim[0], stride=1, dropout_p=dropout_p)] \
              +[FusedMBConv3d(in_dim=hidden_dim[0], out_dim=out_dim, stride=1, dropout_p=dropout_p) for _ in range(1) ]
        )

    def forward(self, enc_out, stage_outputs):

        h1 = self.up_conv1(enc_out)
        h1 = torch.concat([h1, stage_outputs[-1]], dim=1)
        h1 = self.conv1(h1)

        h2 = self.up_conv2(h1)
        h2 = torch.concat([h2, stage_outputs[-2]], dim=1)
        h2 = self.conv2(h2)

        h3 = self.up_conv3(h2)
        h3 = torch.concat([h3, stage_outputs[-3]], dim=1)
        h3 = self.conv3(h3)

        h4 = self.up_conv4(h3)
        h4 = self.conv4(h4)

        return h4

    

