import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .training import training_main
from .mlp_head import Psd_Pred_MLP_Head

from src.setup import embeddings_path, device, feat_dim, model_weights_path
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image


print('-Loading embeddings...')

_IMAGE_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
IMAGE_EMBEDDINGS=[]
for e in _IMAGE_EMBEDDINGS:
    IMAGE_EMBEDDINGS.extend(e)

print('...done loading embeddings (check memory usage)')

image_size = IMAGE_EMBEDDINGS[0].shape[0]

assert IMAGE_EMBEDDINGS[0].shape[-1] == feat_dim

file_names, _ = zip(*get_fnames())

DATA = list(zip(file_names, IMAGE_EMBEDDINGS))

def segmented_image_generator():
    
    print('-Loading model weights...')
    classification_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
    classification_head.load_state_dict(torch.load(os.path.join(model_weights_path, 'psd_head_weights.pt')))
    classification_head.eval()
    print('...done loading model')
    
    for file_name, image in tqdm(DATA, total=len(DATA), desc='Segmenting', leave=False):
        flattened_image = image.reshape(-1, feat_dim).to(device)
        with torch.no_grad():
            predictions = classification_head(flattened_image)
            predictions = torch.round(predictions).cpu().numpy()

        mask = np.array(predictions).reshape(image_size, image_size)
        
        h, w = mask.shape
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

        initial_image = load_image(file_name)[0]
        resized_image = resize_hdf_image(initial_image)

        ax.imshow(resized_image, cmap='gray', extent=[0, h, w, 0])

        # Overlay the heatmap
        sns.heatmap(
            mask,
            cmap='grey',
            alpha=0.5,             # Make the heatmap semi-transparent
            ax=ax,
            cbar=False,
            xticklabels=False,
            yticklabels=False
        )

        plt.title("Segmentation")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    training_main()

    segmented_image_generator()