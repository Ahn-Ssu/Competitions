import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes=13):
        super(BaseModel, self).__init__()
        self.feature_extract = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
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


class efficientNet3D(nn.Module):
    def __init__(self, num_classes=13) -> None:
        super(efficientNet3D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            MBConv(in_dim=32, hidden_dim=32, out_dim=16, kernel_size=3, stride=1, padding=1, scale=False),
            StochasticDepth(p=0.0, mode='row')
        )

        self.conv3 = nn.Sequential(
            MBConv(in_dim=16, hidden_dim=96, out_dim=24, kernel_size=3, stride=2, padding=1, scale=True),
            StochasticDepth(p=0.0125, mode='row'),
            MBConv(in_dim=24, hidden_dim=144, out_dim=24, kernel_size=3, stride=1, padding=1, scale=True),
            StochasticDepth(p=0.025, mode='row')
        )

        self.conv4 = nn.Sequential(
            MBConv(in_dim=24, hidden_dim=144, out_dim=40, kernel_size=5, stride=2, padding=2, scale=True),
            StochasticDepth(p=0.0375, mode='row'),
            MBConv(in_dim=40, hidden_dim=240, out_dim=40, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.05, mode='row'),
        )

        self.conv5 = nn.Sequential(
            MBConv(in_dim=40, hidden_dim=240, out_dim=80, kernel_size=3, stride=2, padding=1, scale=True),
            StochasticDepth(p=0.0625, mode='row'),
            MBConv(in_dim=80, hidden_dim=480, out_dim=80, kernel_size=3, stride=1, padding=1, scale=True),
            StochasticDepth(p=0.075, mode='row'),
            MBConv(in_dim=80, hidden_dim=480, out_dim=80, kernel_size=3, stride=1, padding=1, scale=True),
            StochasticDepth(p=0.0875, mode='row'),
        )

        self.conv6 = nn.Sequential(
            MBConv(in_dim=80, hidden_dim=480, out_dim=112, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.1, mode='row'),
            MBConv(in_dim=112, hidden_dim=672, out_dim=112, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.1125, mode='row'),
            MBConv(in_dim=112, hidden_dim=672, out_dim=112, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.125, mode='row'),
        )

        self.conv7 = nn.Sequential(
            MBConv(in_dim=112, hidden_dim=672, out_dim=192, kernel_size=5, stride=2, padding=2, scale=True),
            StochasticDepth(p=0.1375, mode='row'),
            MBConv(in_dim=192, hidden_dim=1152, out_dim=192, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.15, mode='row'),
            MBConv(in_dim=192, hidden_dim=1152, out_dim=192, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.1625, mode='row'),
            MBConv(in_dim=192, hidden_dim=1152, out_dim=192, kernel_size=5, stride=1, padding=2, scale=True),
            StochasticDepth(p=0.175, mode='row'),
        )

        self.conv8 = nn.Sequential(
            MBConv(in_dim=192, hidden_dim=1152, out_dim=320, kernel_size=3, stride=1, padding=1, scale=False),
            StochasticDepth(p=0.1875, mode='row')
        )

        self.conv9 = nn.Sequential(
            nn.Conv3d(320, 1280, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1280),
            nn.SiLU()
        )

        self.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.drop = nn.Dropout(p=0.2)
        self.clf  = nn.Linear(1280, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        p = self.pool(x)
        # print(p.shape)
        # p = Rearrange(p, 'b ')
        # p = Rearrange(p, ' c t w h -> (c t w h)')
        p = p.view(x.shape[0], -1)
        # [4, 1280, 1, 1, 1]

        out = self.clf(p)

        return out

        

class SqueezeExcitation(nn.Module):
    def __init__(self, in_dim, sqz_dim) -> None:
        super(SqueezeExcitation, self).__init__()

        self.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc1 = nn.Conv3d(in_dim, sqz_dim, kernel_size=1, stride=1)
        self.fc2 = nn.Conv3d(sqz_dim, in_dim, kernel_size=1, stride=1)
        self.act = nn.SiLU()
        self.scale_act = nn.Sigmoid()

    
    def forward(self, x):

        squeezed = self.pool(x)

        e = self.fc1(squeezed)
        e = self.act(e)
        e = self.fc2(e)
        e = self.scale_act(e)

        out = x * e
        
        return out

class MBConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size, stride, padding, scale=True) -> None:
        super(MBConv, self).__init__()

        self.scale = scale

        if self.scale:
            self.bottleneck = nn.Sequential(
                nn.Conv3d(in_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU()
            )

            self.conv1 = nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU()
            )
        
        self.SqueezeExcitation = SqueezeExcitation(hidden_dim, 8 if hidden_dim == 32 else hidden_dim//24)

        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_dim)
        )
    
    def forward(self, x):

        if self.scale:
            x = self.bottleneck(x)
            
        h = self.conv1(x)
        h = self.SqueezeExcitation(h)
        h = self.conv2(h)

        return h

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