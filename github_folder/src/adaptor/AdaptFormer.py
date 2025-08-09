import torch
from torch import nn
import numpy as np

class AdaptMLP(nn.Module):
    def __init__(self, device, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1): # TODO: CHANGE DEPENDING ON TASK
        super().__init__()
        
        self.device = device
        self.original_mlp = original_mlp # original MLP block
        
        # down --> non linear --> up
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s # scaling factor
        
        # initialization
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        
        # freeze original MLP
        for p in self.original_mlp.parameters():
            p.requires_grad = False
        
        self.to(self.device)

    def forward(self, x):

        down = self.down_proj(x)
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)

        output = self.original_mlp(x) + up * self.scale

        return output