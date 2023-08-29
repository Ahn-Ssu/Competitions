# molecule predictor 
import torch
import torch.nn as nn 
import torch.nn.functional as F


from model.GNN_encoder import GIN


class Molp(nn.Module):
    def __init__(self, num_atom_f, gnn_hidden_dims, num_mol_f, d_model, FF_hidden_dim, output_dim, dropout_p) -> None:
        super(Molp, self).__init__()

        self.gnn_encoder = GIN(input_dim=num_atom_f, hidden_dims=gnn_hidden_dims)
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=8,
                                          dim_feedforward=FF_hidden_dim,
                                          dropout=dropout_p,
                                          activation=F.selu)
        
        self.projection = nn.Linear(in_features=d_model, out_features=output_dim)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input):
        enc_out = self.gnn_encoder(input)

        mol_f = input.mol_f
        BZ, _ = mol_f.size()

        h = torch.concat([enc_out, mol_f], dim=1) # BZ, 512 + 40 
        h = h.view(BZ, 1, -1)
        h = self.transformer(h, h)
        h = self.norm(h)
        h = h.view(BZ, -1)
        h = self.projection(h)

        return h
        
        


