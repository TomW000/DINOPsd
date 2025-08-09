import os
import json
from glob import glob
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ---------------------------------------------------------------------------------------------------------------------
# Parameters to tune:
# ---------------------------------------------------------------------------------------------------------------------

dataset_path = None
ground_truth_path = None

# select model size
model_size = 'small' #@param {type:"string", options:["small", "base", "large", "giant"]}

# select image size - this is going to influence the realtive patch size
resize_size = 518 
# select upsampling method
upsample = "bilinear" #@param {type:"string", options:["bilinear", "Nearest Neighbor", "None"], value-map:{bilinear:"bilinear", "Nearest Neighbor": "nearest", None:None}}

# =====================================================================================================================

# ---------------------------------------------------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'cpu')
print("Device:", device)

model_dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
assert model_size in model_dims, f'Invalid model size: ({model_size})'
model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size[0]}14_reg')
model.to(device) # type: ignore
model.eval() # type: ignore

feat_dim = model_dims[model_size]
patch_size = model.patch_size # type: ignore
print("Model loaded")

# ---------------------------------------------------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------------------------------------------------

directory_path = os.getcwd()
dir_names = ['output', 'embeddings', 'model_weights']
config_file = os.path.join(directory_path, "directories.json")

# Load previous sub_directories if file exists
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        sub_directories = json.load(f)
else:
    sub_directories = {}

# Create directories if missing
for sub_directory in dir_names:
    target = os.path.join(directory_path, sub_directory)
    if not os.path.exists(target):
        try:
            os.makedirs(target, exist_ok=True)
            print(f"Directory {sub_directory} created")
        except PermissionError:
            print(f"Permission denied for {sub_directory}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    # Always store the path in sub_directories
    sub_directories[sub_directory] = target
    print("Directory map saved to directories.json")

# Save updated mapping to file
with open(config_file, 'w') as f:
    json.dump(sub_directories, f, indent=2)

# Update directory paths
output_directory = sub_directories['output']
embeddings_directory = sub_directories['embeddings']
model_weights_directory = sub_directories['model_weights']
