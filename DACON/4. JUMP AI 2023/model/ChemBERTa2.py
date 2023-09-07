# molecule predictor 
import torch
import torch.nn as nn 
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import chem



class ChemBERT(nn.Module):
    def __init__(self, out_dim) -> None:
        super(ChemBERT, self).__init__()
        self.ChemBERT_encoder = AutoModelForSequenceClassification.from_pretrained(
            chem.chosen, num_labels=out_dim, problem_type="multi_label_classification"
        )

    def forward(self, batch):
        logits = self.ChemBERT_encoder(batch.input_ids).logits
        return logits

if __name__ == '__main__':
    model = ChemBERT(out_dim=2)
    print(model)