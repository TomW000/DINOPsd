import numpy as np
from torch import nn

def init_model(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class Psd_Pred_MLP_Head(nn.Module):
    def __init__(self, device, nb_outputs,feat_dim, use_sigmoid=False):
        super().__init__()
        self.device = device
        self.nb_outputs = nb_outputs
        self.feat_dim = feat_dim
        self.hidden_dims = self.feat_dim*np.array([1.5, 1, 3/4, 1/2, 1/4])
        self.hidden_dims = self.hidden_dims.astype(int)
        self.stack = nn.Sequential(nn.Linear(self.feat_dim,
                                             self.hidden_dims[0]),
                                   nn.ReLU())
        k=1
        while k < len(self.hidden_dims):
            self.stack.append(nn.Linear(self.hidden_dims[k-1], self.hidden_dims[k]))
            self.stack.append(nn.ReLU())
            k+=1
        self.stack.append(nn.Linear(self.hidden_dims[-1], self.nb_outputs))

        if use_sigmoid:
            self.stack.append(nn.Sigmoid())

        self.apply(init_model)
        self.to(self.device)

    def forward(self, x):
        return self.stack(x)