import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .training import main
from .head import Psd_Pred_MLP_Head

from src.setup import embeddings_path, device, feat_dim, model_weights_path
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image


if __name__ == '__main__':

    main() 

    classification_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
    classification_head.load_state_dict(torch.load(os.path.join(model_weights_path, 'psd_head_weights.pt')))
    classification_head.eval()

    print('Loading embeddings...')

    IMAGE_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
    IMAGE_EMBEDDINGS = torch.tensor(np.vstack(IMAGE_EMBEDDINGS))
    
    print('...done loading embeddings (check memory usage)')
    
    image_size = IMAGE_EMBEDDINGS.shape[1]

    file_names, _ = zip(*get_fnames())

    DATA = list(zip(file_names, IMAGE_EMBEDDINGS))
    
    def segmented_image_generator():
        for file_name, image in tqdm(DATA, total=len(DATA), desc='Segmenting', leave=False):
            flattened_image = image.reshape(-1, feat_dim)
            flattened_image = flattened_image.to(device)
            image_prediction = []
            for patch in tqdm(flattened_image, desc='Predicting', leave=False):
                with torch.no_grad():
                    prediction = classification_head(patch)
                    print(prediction)
                    prediction = torch.argmax(prediction).cpu().numpy()
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
            
    mask_generator = segmented_image_generator()

    for mask in mask_generator:
        plt.imshow(mask, cmap='gray')
        plt.show()