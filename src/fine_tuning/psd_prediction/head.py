import numpy as np
from torch import nn

from src.setup import device, feat_dim

def init_model(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class Psd_Pred_MLP_Head(nn.Module):
    def __init__(self, device, feat_dim):
        super().__init__()
        self.device = device
        self.nb_outputs = 1
        self.feat_dim = feat_dim
        self.hidden_dims = self.feat_dim*np.array([3/4, 1/2, 1/4])
        self.hidden_dims = self.hidden_dims.astype(int)

        self.stack = nn.Sequential(nn.Linear(self.feat_dim,
                                             self.hidden_dims[0]),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dims[0],
                                             self.hidden_dims[1]),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dims[1],
                                             self.hidden_dims[2]),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dims[2],
                                             self.nb_outputs),
                                   nn.Sigmoid()) #FIXME: dim?

        self.apply(init_model)
        self.to(self.device)

    def forward(self, x):
        return self.stack(x)
    
    
detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
detection_head.to(device)