import torch
import torch.nn as nn
import torchvision.models as models
# from inceptionNeXt import inceptionnext_tiny
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
class BaseModel(nn.Module):
    def __init__(self, num_classes:int):
        super(BaseModel, self).__init__()
        
        # self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        self.backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        # self.backbone = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-384-224-1k")
        # self.backbone = inceptionnext_tiny(pretrained=True)
        # self.backbone = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        # self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
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