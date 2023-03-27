import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
import torchvision.models as models


# https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth
# https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth
# https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth
class BaseModel(nn.Module):
    def __init__(self, num_classes=13, args=None):
        super(BaseModel, self).__init__()


        if args.model_name == 'r3d_18':
            self.feature_extract = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
        elif args.model_name == 'r2plus1d_18':
            self.feature_extract = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)
        else:
            self.feature_extract = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
                
        self.drop = nn.Dropout(p=0.1)
        self.act  = nn.SiLU()
        self.classifier = nn.Linear(400, num_classes)
        
    def forward(self, x):
        # batch_size = x.size(0)
        x = self.feature_extract(x)
        # x = x.view(batch_size, -1)
        x = self.drop(x)
        x = self.act(x)
        x = self.classifier(x)
        return x

class givenModel(nn.Module):
    def __init__(self, num_classes=13):
        super(givenModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((3, 7, 7)),
        )
        self.classifier = nn.Linear(1024, num_classes)
        
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
    

class SqueezeExcitation3D(nn.Module):
    def __init__(self, in_dim, reduction_ratio=4) -> None:
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)

        self.excitation = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=in_dim//reduction_ratio, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Conv3d(in_channels=in_dim//reduction_ratio, out_channels=in_dim, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        se_out = self.squeeze(x)
        se_out = self.excitation(se_out)
        se_out = se_out * x

        return se_out
    
class FusedMBConv3D(nn.Module):
    def __init__(self, in_dim, out_dim, expansion_ratio=4, squeeze_ratio=4, kernel_size=3, stride=1) -> None:
        super(FusedMBConv3D, self).__init__()

        self.use_residual = in_dim == out_dim and stride == 1
        hidden_dim = int(in_dim * expansion_ratio)
        padding = kernel_size // 2 

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        )

        self.se = SqueezeExcitation3D(hidden_dim, reduction_ratio=squeeze_ratio)

        self.projectiion = nn.Sequential(
            nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_dim),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):

        h = self.conv(x)
        h = self.se(h)
        h = self.projectiion(h)

        if self.use_residual:
            h = h + x
        
        return h

class R3DcpSE(nn.Module):
    def __init__(self, num_classes=13) -> None:
        super(R3DcpSE, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.SiLU()
        )

        self.conv1 = nn.Sequential(
            *([FusedMBConv3D(64, 64, 4, 4, 3, 2)]
              +[FusedMBConv3D(64, 64, 4, 4, 3, 1) for _ in range(3)])
        )

        self.conv2 = nn.Sequential(
            *([FusedMBConv3D(64, 128, 4, 4, 3, 2)]
              +[FusedMBConv3D(128, 128, 4, 4, 3, 1) for _ in range(4)])
        )

        self.conv3 = nn.Sequential(
            *([FusedMBConv3D(128, 256, 4, 4, 3, 2)]
              +[FusedMBConv3D(256, 256, 4, 4, 3, 1) for _ in range(5)])
        )

        self.conv4 = nn.Sequential(
            *([FusedMBConv3D(256, 512, 4, 4, 3, 2)]
              +[FusedMBConv3D(512, 512, 4, 4, 3, 1) for _ in range(5)])
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Conv3d(in_channels=512, out_channels=num_classes)

    def forward(self, x):

        h = self.stem(x)

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        p = self.avgpool(h)
        p = self.dropout(p)

        out = self.fc(p)
        out = out.view(out.size(0), -1)

        return out
        


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        


