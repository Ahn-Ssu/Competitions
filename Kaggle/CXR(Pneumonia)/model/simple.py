import torch
import torch.nn as nn 
import torch.nn.functional as F

import timm

class simple_CNN(nn.Module):
    def __init__(self) -> None:
        super(simple_CNN, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.backbone = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k',
                                            pretrained=True,
                                            num_classes=0
                                            # features_only=True,
                                        )
        # 1280로 나옵니당
        self.classifier = nn.Linear(in_features=1280, out_features=2)
        
    def forward(self, inputs):
        
        h = self.conv(inputs)
        h = self.backbone(h)
        h = self.classifier(h)
        h = F.softmax(h, dim=1)
        # h = F.sigmoid(h)
        # h = h.squeeze()
        
        return h
        
        

if __name__ == '__main__':
    
    model = simple_CNN()
    
    inputs = torch.rand(1,1,512,512)
    
    inputs = inputs.to('cuda:0')
    model = model.to('cuda:0')
    
    ret = model(inputs)