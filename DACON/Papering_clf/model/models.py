import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes:int):
        super(BaseModel, self).__init__()
        
        self.backbone = models.convnext_tiny(pretrained=True)
        self.norm = nn.LayerNorm(1000)
        self.act = nn.SiLU()
        self.drop = nn.Dropout1d()
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x