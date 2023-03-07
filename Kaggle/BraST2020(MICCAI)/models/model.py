import torch.nn as nn
import torch 
import torch.nn.functional as F


from easydict import EasyDict 

class VNet(nn.Module):
    def __init__(self, args=None) -> None:
        super(VNet, self).__init__()

        self.encoder = CompressionPath(args)
        self.decoder = DecompressionPath(args)

    
    def forward(self, x, activate=False):

        enc_out, stage_outputs = self.encoder(x)
        dec_out = self.decoder(enc_out, stage_outputs)


        if not activate:
            return dec_out
        else:
            output = F.softmax(dec_out, dim=1)
        return output

# Encoder; 'compression path'
# ... The left part of the network consists of a compress, ...
class CompressionPath(nn.Module):
    def __init__(self, args=None) -> None:
        super(CompressionPath, self).__init__()

        if args is None:
            args = EasyDict()
            args.in_dim  = 4
            args.net_dim = [16, 32, 64, 128, 256]

            conv_kwargs = EasyDict()
            conv_kwargs.kernel_size = (5,5,5)
            conv_kwargs.stride      = (1,1,1)
            conv_kwargs.padding     = 2

            down_kwargs = EasyDict()
            down_kwargs.kernel_size = (2,2,2)
            down_kwargs.stride      = (2,2,2)
            down_kwargs.padding     = 0

            args.conv_kwargs = conv_kwargs
            args.down_kwargs = down_kwargs

        self.conv1 = nn.Sequential(
                nn.Conv3d(args.in_dim, args.net_dim[0], kernel_size=5, stride=1, padding=2),
                nn.SiLU()
            )
        self.down1 = nn.Sequential(
                nn.Conv3d(args.net_dim[0], args.net_dim[1], **args.down_kwargs),
                nn.SiLU()
            )

        self.conv2 = nn.Sequential(
                nn.Conv3d(args.net_dim[1], args.net_dim[1], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[1], args.net_dim[1], **args.conv_kwargs),
                nn.SiLU()
            )
        self.down2 = nn.Sequential(
                nn.Conv3d(args.net_dim[1], args.net_dim[2], **args.down_kwargs),
                nn.SiLU()
            )
        
        self.conv3 = nn.Sequential(
                nn.Conv3d(args.net_dim[2], args.net_dim[2], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[2], args.net_dim[2], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[2], args.net_dim[2], **args.conv_kwargs),
                nn.SiLU()
            )
        self.down3 = nn.Sequential(
                nn.Conv3d(args.net_dim[2], args.net_dim[3], **args.down_kwargs),
                nn.SiLU()
            )

        self.conv4 = nn.Sequential(
                nn.Conv3d(args.net_dim[3], args.net_dim[3], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[3], args.net_dim[3], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[3], args.net_dim[3], **args.conv_kwargs),
                nn.SiLU()
            )
        self.down4 = nn.Sequential(
                nn.Conv3d(args.net_dim[3], args.net_dim[4], **args.down_kwargs),
                nn.SiLU()
            )

        self.conv5 = nn.Sequential(
                nn.Conv3d(args.net_dim[4], args.net_dim[4], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[4], args.net_dim[4], **args.conv_kwargs),
                nn.SiLU(),
                nn.Conv3d(args.net_dim[4], args.net_dim[4], **args.conv_kwargs),
                nn.SiLU()
            )

        

    def forward(self, x) :
        
        h1  = self.conv1(x)
        d1  = self.down1(h1)

        id2 = d1
        h2  = self.conv2(d1)
        h2  = h2 + id2
        d2  = self.down2(h2)

        id3 = d2 
        h3  = self.conv3(d2)
        h3  = h3 + id3 
        d3  = self.down3(h3)

        id4 = d3 
        h4  = self.conv4(d3)
        h4  = h4 + id4
        d4  = self.down4(h4)

        id5 = d4
        h5  = self.conv5(d4)
        h5  = h5 + id5

        stage_outputs = [h1, h2, h3, h4] # forward the features extracted from early stages of the left part of the CNN to the right part

        return h5, stage_outputs

# decoder; 'decompression path'
# while the right part decompresses the signal until its original size is reached.
class DecompressionPath(nn.Module):
    def __init__(self, args=None) -> None:
        super(DecompressionPath, self).__init__()

        if args is None:
            args = EasyDict()
            args.out_dim = 2
            args.net_dim = [16, 32, 64, 128, 256]

            conv_kwargs = EasyDict()
            conv_kwargs.kernel_size = (5,5,5)
            conv_kwargs.stride      = (1,1,1)
            conv_kwargs.padding     = 2

            up_kwargs = EasyDict()
            up_kwargs.kernel_size = (2,2,2)
            up_kwargs.stride      = (2,2,2)
            up_kwargs.padding     = 0

            args.conv_kwargs = conv_kwargs
            args.up_kwargs   = up_kwargs
            

        self.up1 = nn.Sequential(
                nn.ConvTranspose3d(args.net_dim[-1], args.net_dim[-2], **args.up_kwargs),
                nn.PReLU()
            )
        self.conv1 = nn.Sequential(
                nn.Conv3d(args.net_dim[-1], args.net_dim[-1], **args.conv_kwargs),
                nn.PReLU(),
                nn.Conv3d(args.net_dim[-1], args.net_dim[-1], **args.conv_kwargs),
                nn.PReLU(),
                nn.Conv3d(args.net_dim[-1], args.net_dim[-1], **args.conv_kwargs),
                nn.PReLU()
            )

        self.up2 = nn.Sequential(
                nn.ConvTranspose3d(args.net_dim[-2], args.net_dim[-3], **args.up_kwargs),
                nn.PReLU()
            )
        self.conv2 = nn.Sequential(
                nn.Conv3d(args.net_dim[-2], args.net_dim[-2], **args.conv_kwargs),
                nn.PReLU(),
                nn.Conv3d(args.net_dim[-2], args.net_dim[-2], **args.conv_kwargs),
                nn.PReLU(),
                nn.Conv3d(args.net_dim[-2], args.net_dim[-2], **args.conv_kwargs),
                nn.PReLU()
            )
        
        self.up3 = nn.Sequential(
                nn.ConvTranspose3d(args.net_dim[-3], args.net_dim[-4], **args.up_kwargs),
                nn.PReLU()
            )
        self.conv3 = nn.Sequential(
                nn.Conv3d(args.net_dim[-3], args.net_dim[-3], **args.conv_kwargs),
                nn.PReLU(),
                nn.Conv3d(args.net_dim[-3], args.net_dim[-3], **args.conv_kwargs),
                nn.PReLU()
            )

        self.up4 = nn.Sequential(
                nn.ConvTranspose3d(args.net_dim[-4], args.net_dim[-5], **args.up_kwargs),
                nn.PReLU()
            )
        self.conv4 = nn.Sequential(
                nn.Conv3d(args.net_dim[-4], args.net_dim[-4], **args.conv_kwargs),
                nn.PReLU()
            )

        self.conv5 = nn.Sequential(
                nn.Conv3d(args.net_dim[-4], args.out_dim, kernel_size=1, stride=1, padding=0),
                nn.PReLU()
            )

    def forward(self, enc_out, stage_outputs):
        
            u1  = self.up1(enc_out)
            id1 = u1
            h1  = torch.cat([u1, stage_outputs[-1]], dim=1)
            h1  = self.conv1(h1)

            u2  = self.up2(h1)
            id2 = u2
            h2  = torch.cat([u2, stage_outputs[-2]], dim=1)
            h2  = self.conv2(h2)
            h2  = h2 + id2

            u3  = self.up3(h2)
            id3 = u3
            h3  = torch.cat([u3, stage_outputs[-3]], dim=1)
            h3  = self.conv3(h3)
            h3  = h3 + id3

            u4  = self.up4(h3)
            id4 = u4
            h4  = torch.cat([u4, stage_outputs[-4]], dim=1)
            h4  = self.conv4(h4)
            h4  = h4 + id4

            out = self.conv5(h4)

            return out