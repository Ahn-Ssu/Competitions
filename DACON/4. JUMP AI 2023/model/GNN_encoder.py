import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as geonn

# geonn.TransformerConv, SAGEConv 

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, expand_rate=4) -> None:
        super().__init__()
        
        self.layer1 = nn.Linear(input_dim, output_dim*expand_rate, bias=False)
        self.norm   = nn.BatchNorm1d(output_dim*expand_rate)
        self.act    = nn.SELU()
        self.layer2 = nn.Linear(output_dim*expand_rate, output_dim)
        
        self._init_layer_weights()
        
    def _init_layer_weights(self):
        for module in self.modules():
            if hasattr(module, 'weights'):
                nn.init.kaiming_normal_(module.weight, nonlinearity='selu')  # relu, leaky_relu, selu

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.layer2(x)
        
        return x
        

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(GIN, self).__init__()

        self.conv1 = geonn.GINConv(
            MLP(input_dim=input_dim, output_dim=hidden_dims[0])
        )
        self.conv2 = geonn.GINConv(
            MLP(hidden_dims[0], hidden_dims[1])
        )
        self.conv3 = geonn.GINConv(
            MLP(hidden_dims[1], hidden_dims[2])
        )
        self.conv4 = geonn.GINConv(
            MLP(hidden_dims[2], hidden_dims[3])
        )
        self.norm1 = geonn.GraphNorm(hidden_dims[0])
        self.norm2 = geonn.GraphNorm(hidden_dims[1])
        self.norm3 = geonn.GraphNorm(hidden_dims[2])
        self.norm4 = geonn.GraphNorm(hidden_dims[3])

    def forward(self, g):

        h1 = self.conv1(g.x, g.edge_index, g.edge_attr)
        h1 = self.norm1(h1, g.batch)
        h1 = self.act(h1)

        h2 = self.conv2(h1, g.edge_index, g.edge_attr)
        h2 = self.norm2(h2, g.batch)
        h2 = self.act(h2)

        h3 = self.conv3(h2, g.edge_index, g.edge_attr)
        h3 = self.norm3(h3, g.batch)
        h3 = self.act(h3)


        h4 = self.conv4(h3, g.edge_index, g.edge_attr)
        h4 = self.norm4(h4, g.batch)
        h4 = self.act(h4)

        hg = torch.cat([h1, h2, h3, h4],
                       dim=1)
        # Read Out Layer 
        hg = geonn.global_mean_pool(hg, g.batch)

        return hg