import torch
import torch.nn as nn 
import torch.nn.functional as F

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sin(x)
    
    
class MLP_VAE(nn.Module):
    def __init__(self, data_dim):
        super(MLP_VAE,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, 32),
            nn.BatchNorm1d(32),
            nn.Dropout1d(0.125),
            SinActivation(),   
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Dropout1d(0.125 * 2),
            SinActivation(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Dropout1d(0.125 * 3),
            SinActivation(),
        )
        
        self.fc_mu = nn.Linear(128, 10)
        self.fc_var = nn.Linear(128, 10)
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.BatchNorm1d(128),
            nn.Dropout1d(0.125 * 2),
            SinActivation(),
            nn.Linear(128, 64),
            nn.Dropout1d(0.125 * 1),
            nn.BatchNorm1d(64),
            SinActivation(),
            nn.Linear(64, data_dim),
        )
                
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        recon = self.decoder(z)
        return recon
    
                
    def forward(self, cat_f, num_f):                # x: (batch_size, 1, 28, 28)
        features = torch.concat([cat_f, num_f], dim=1)
        mu, log_var = self.encode(features)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var
    
    
    
if __name__ == "__main__":
    #MSE = nn.MSELoss()
    CE = torch.nn.CrossEntropyLoss(reduction='sum')
    MSE = torch.nn.MSELoss(reduction='sum')
    

    def loss_func(cat_f, num_f, recon_x, mu, log_var):
        #batch_size = x.size(0)
        #MSE_loss = MSE(x, recon_x.view(batch_size, 1, 28, 28))
        
        recon_cat = recon_x[..., :4]
        recon_num = recon_x[..., 4:]
        

        CE_loss = CE(recon_cat, cat_f)
        MSE_loss = MSE(recon_num, num_f)
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return CE_loss + MSE_loss + KLD_loss 

    model = MLP_VAE(data_dim=21)
    
    cat_f = torch.rand((10, 4))
    num_f = torch.rand((10, 17))
    
    outputs, mu, log_var = model(cat_f, num_f)
    print(f'{outputs.shape=}')
    print(f'{mu.shape=}')
    print(f'{log_var.shape=}')
    loss = loss_func(cat_f, num_f, outputs, mu, log_var)
    print(loss)
    
    print(f'{torch.concat([mu, mu], dim=0).shape}')