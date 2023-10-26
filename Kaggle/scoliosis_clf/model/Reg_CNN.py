import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models
import timm

class simple_CNN(nn.Module):
    def __init__(self) -> None:
        super(simple_CNN, self).__init__()
        
        self.backbone = backbone_pt()
        self.classifier = nn.Linear(in_features=1280, out_features=1)

        self.load_weights()

    def load_weights(self):
        self.backbone.load_state_dict(torch.load('/home/pwrai/userarea/spineTeam/model/weights/efficientNet_v2_m.ckpt'))
        
    def forward(self, inputs):
        
        h = self.backbone(inputs)
        h = self.classifier(h)
        # h = F.sigmoid(h)
        h = F.softplus(h)
        
        return h
        
class backbone_pt(nn.Module):
    def __init__(self) -> None:
        super(backbone_pt, self).__init__()        

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.backbone = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k',
                                            pretrained=False,
                                            num_classes=0,
                                            #features_only=True,
                                      )

    def forward(self, inputs):
        h = self.conv(inputs)
        return self.backbone(h)
if __name__ == '__main__':
    
    model = simple_CNN()
    
    inputs = torch.rand(4,1,512,512)
    
    inputs = inputs.to('cuda:0')
    model = model.to('cuda:0')
    
    ret = model(inputs)
    print(ret.shape)
    
