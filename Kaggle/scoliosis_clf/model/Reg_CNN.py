import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models
import timm

class simple_CNN(nn.Module):
    def __init__(self) -> None:
        super(simple_CNN, self).__init__()
        
        self.backbone = backbone_pt()
#         self.backbone = models.resnet34()
        self.classifier = nn.Linear(in_features=1000, out_features=2)

#         self.load_weights()

#     def load_weights(self):
#         self.backbone.load_state_dict(torch.load('/home/pwrai/userarea/spineTeam/model/weights/backbone_efficient_v2_m.ckpt'))
#         self.backbone.out_norm = nn.BatchNorm1d(1280)

    def forward(self, inputs):
        h = self.backbone(inputs)
        h = self.classifier(h)
        # h = F.sigmoid(h)
#         h = F.softplus(h)
#         h = torch.abs(h)
        return h
        
class backbone_pt(nn.Module):
    def __init__(self) -> None:
        super(backbone_pt, self).__init__()        

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1,bias=False)
        self.norm = nn.BatchNorm2d(num_features=3)
        self.out_norm = nn.BatchNorm1d(num_features=1000)
        self.act = torch.sin
#         self.backbone = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k',
#                                             pretrained=False,
#                                             num_classes=0,
#                                             #features_only=True,
#                                       )
        self.backbone = models.resnet34()
    def forward(self, inputs):
        h = self.conv(inputs)
        h = self.norm(h)
        h = self.act(h)
        h = self.backbone(h)
        h = self.out_norm(h)
        h = self.act(h)
        return h
    
if __name__ == '__main__':
    
    model = simple_CNN()
    
    inputs = torch.rand(1,512,512)
    
    inputs = inputs.to('cuda:0')
    model = model.to('cuda:0')
    model.eval()
    
    ret = model(inputs)
    print(ret.shape)
    
    angle = torch.atan2(ret[..., 0], ret[..., 1]) * 180 / 3.1415
    print(angle.shape)
    
