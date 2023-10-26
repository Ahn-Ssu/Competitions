import torch
import torch.nn as nn

class SMAPELoss(nn.Module):
    def __init__(self) -> None:
        super(SMAPELoss, self).__init__()
        
        self.l1_loss = nn.L1Loss()
        self.epsilon = 1e-6
        
    def forward(self, inputs, targets):
        
        l1loss = self.l1_loss(inputs, targets)
        y = torch.sum(torch.abs(inputs))
        y_hat = torch.sum(torch.abs(targets))
        
        denominator = y + y_hat + self.epsilon
        
        return (l1loss / denominator)
    


def RMSSE(true, pred, train): 
    '''
    true: np.array 
    pred: np.array
    train: np.array
    '''
    
    n = len(train)

    numerator = np.mean(np.sum(np.square(true - pred)))
    
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    
    msse = numerator/denominator
    
    return msse ** 0.5

def SMAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) #100은 상수이므로 이번 코드에서는 제외

def MAE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs(true-pred))

if __name__ =='__main__':
    import numpy as np 
    
    l1loss = nn.L1Loss()
    smapeloss = SMAPELoss()
    
    TRUE_UNDER = np.array([10, 20, 30, 40, 50])
    PRED_OVER = np.array([30, 40, 50, 60, 70])
    TRUE_OVER = np.array([30, 40, 50, 60, 70])
    PRED_UNDER = np.array([10, 20, 30, 40, 50])
    
    print(MAE(PRED_OVER, TRUE_UNDER))
    print(SMAPE(PRED_OVER, TRUE_UNDER))
    
    print(MAE(PRED_UNDER, TRUE_OVER))
    print(SMAPE(PRED_UNDER, TRUE_OVER))
    
    