import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .head import Psd_Pred_MLP_Head

from src.setup import embeddings_path, device, feat_dim, model_weights_path
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image


classification_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
classification_head.load_state_dict(torch.load(os.path.join(model_weights_path, 'psd_head_weights.pt')))
classification_head.eval()

print('Loading embeddings...')

_IMAGE_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
IMAGE_EMBEDDINGS=[]
for e in _IMAGE_EMBEDDINGS:
    IMAGE_EMBEDDINGS.extend(e)

print('...done loading embeddings (check memory usage)')

image_size = IMAGE_EMBEDDINGS[0].shape[0]

file_names, _ = zip(*get_fnames())

DATA = list(zip(file_names, IMAGE_EMBEDDINGS))

def segmented_image_generator():
    for file_name, image in tqdm(DATA[:600], total=len(DATA), desc='Segmenting', leave=False):
        flattened_image = image.reshape(-1, feat_dim)
        flattened_image = flattened_image.to(device)
        image_prediction = []
        for patch in tqdm(flattened_image, desc='Predicting', leave=False):
            with torch.no_grad():
                prediction = classification_head(patch)
                prediction = torch.round(prediction).cpu().numpy()
                image_prediction.append(prediction)

        mask = np.array(image_prediction).reshape(image_size, image_size)
        
        yield mask
        
        '''
        
        h, w = mask.shape
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

        initial_image = load_image(file_name)[0]
        resized_image = resize_hdf_image(initial_image)

        ax.imshow(resized_image, cmap='gray', extent=[0, h, w, 0])

        # Overlay the heatmap
        sns.heatmap(
            mask,
            cmap='coolwarm',
            alpha=0.5,             # Make the heatmap semi-transparent
            ax=ax,
            cbar=True,
            xticklabels=False,
            yticklabels=False
        )

        plt.title("Segmentation")
        plt.tight_layout()
        plt.show()
        '''
        
print('Here 2')

mask_generator = segmented_image_generator()


if __name__ == '__main__':

    l = []

    for mask in tqdm(mask_generator, desc='Looping through images', leave=False):
        l.append(mask)