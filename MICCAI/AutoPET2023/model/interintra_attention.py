import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraInter_Attention(nn.Module):
    def __init__(self, d_model, n_head=4) -> None:
        super(IntraInter_Attention, self).__init__()
        # transformer 사용 vs Multihead Attn 사용 비교 필요 
        assert d_model % n_head == 0 

        self.PET_inter = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, activation=F.selu)
        self.CT_inter = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, activation=F.selu)

        self.PET_intra = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, activation=F.selu)
        self.CT_intra = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, activation=F.selu)

    def forward(self, h_pet, h_ct):
        bz, d, *size = h_pet.shape

        h_pet = h_pet.view(bz, d, -1).contiguous().transpose(1,2)
        h_ct = h_ct.view(bz, d, -1).contiguous().transpose(1,2)

        pet_inter = self.PET_inter(h_pet, h_ct)
        ct_inter  = self.CT_inter(h_ct, h_pet)

        pet_intra = self.PET_intra(h_pet, h_pet)
        ct_intra  = self.CT_intra(h_ct, h_ct)

        out = torch.concat([pet_inter, ct_inter, pet_intra, ct_intra], dim=2)
        out = out.transpose(1, 2).view(bz, d*4, size[0], size[1], size[2])

        return out # return shape: bz, d*4, *size_
    

if __name__ == '__main__':

    model = IntraInter_Attention(64)

    h_ct = h_pet = torch.rand((1, 64, 128, 128, 128))

    print(f'{h_ct.shape=}')
    ret = model(h_pet, h_ct)
    print(f'{ret.shape=}')