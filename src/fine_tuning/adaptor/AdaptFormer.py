import torch
from torch import nn
import numpy as np

from ..Neuro_Classification.Neuro_Classification_Head import head 


device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'mps' if torch.mps.is_available()
                      else 'cpu')
print("Device:", device)

# select model size
model_size = 'small' #@param {type:"string", options:["small", "base", "large", "giant"]}

model_dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
assert model_size in model_dims, f'Invalid model size: ({model_size})'
model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size[0]}14_reg')
model.to(device)
model.eval()

feat_dim = model_dims[model_size]


for param in head.parameters():
    param.requires_grad = False
    

for param in model.parameters():
    param.requires_grad = False


class AdaptMLP(nn.Module):
    def __init__(self, device, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1):
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


for k in range(len(list(model.blocks))):

    mlp = model.blocks[k].mlp
    in_dim = model.blocks[k].norm2.normalized_shape[0]
    mid_dim = int(model.blocks[k].norm2.normalized_shape[0]/10) #TODO: important parameter
    
    adapter = AdaptMLP(device, mlp, in_dim, mid_dim)

    model.blocks[k].mlp = adapter


augmented_model = nn.Sequential(model, head)
augmented_model.eval()
augmented_model.to(device)


trainable_params = [p for p in augmented_model.parameters() if p.requires_grad]
params = sum([np.prod(p.size()) for p in trainable_params])

frozen_params_list = [p for p in augmented_model.parameters() if not p.requires_grad]
frozen_params = sum([np.prod(p.size()) for p in frozen_params_list])

total_params = params + frozen_params

print(f'Proportion of trainable parameters: {params / total_params * 100:.2f}%')