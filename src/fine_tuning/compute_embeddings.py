from src.setup import neurotransmitters, model_size, device, feat_dim, resize_size, curated_idx, few_shot_transforms, embeddings_path, model
from src.setup import tqdm, torch, np, os, gc, tqdm
from src.analysis_utils import resize_hdf_image, get_augmented_coordinates
from src.perso_utils import get_fnames, load_image
from src.DinoPsd import DinoPsd_pipeline
from src.DinoPsd_utils import get_img_processing_f


few_shot = DinoPsd_pipeline(model,
                            model.patch_size,
                            device,
                            get_img_processing_f(resize_size),
                            feat_dim, 
                            dino_image_size=resize_size
                            )

files, labels = zip(*get_fnames()) 
resize_factor = resize_size/130
resize = lambda x: resize_factor*x

from src.perso_utils import filter_f
filter = filter_f

indices = [
    [1,3,6,8,9],
    [1,2,4,5,6],
    [0,1,2,4,5],
    [1,2,3,6,7],
    [1,2,5,6,8],
    [0,6,10,11,14]
    ]

coords = [
    [(69,63.5),(68,61),(83,57),(76,62),(60,63)],
    [(66,62),(58.5,64),(64,60),(62.5,65),(64,71)],
    [(65,67),(72,60),(63,72),(60,67),(69,66.5)],
    [(65,66),(64,71),(62,58.5),(62,68),(69,55)],
    [(66,60),(60,70),(61,66.6),(58.5,63.5),(62.5,70.5)],
    [(63,73),(58,69),(60,69),(66,64),(62,71)]
    ]

def compute_ref_embeddings(saved_ref_embeddings=False, 
                           embs_path=None, 
                           k=10,
                           data_aug=False):

    if saved_ref_embeddings:
        
        mean_ref = torch.load(embs_path, weights_only=False)
        return mean_ref

    else:

        if data_aug:    
            nb_transformations = len(few_shot_transforms)
            
            # Preload images and metadata once
            good_images = []
            transformed_coordinates = []

            for idx in curated_idx:
                img, coord_x, coord_y = load_image(files[idx])
                good_images.append(img.transpose(1,2,0))
                transformed_coordinates.append([(0, coord_x, coord_y)] * nb_transformations)

            transformed_images = []
            for image in good_images:
                transformed = [t(image).permute(1,2,0) for t in few_shot_transforms]
                transformed_images.extend(transformed)

            for j, img in enumerate(transformed_images):
                if img.shape != torch.Size([130, 130, 1]):
                    h, w = img.shape[:2]
                    h_diff = (130 - h) // 2
                    w_diff = (130 - w) // 2
                    padded_img = torch.zeros(130, 130, 1)
                    padded_img[h_diff:h+h_diff, w_diff:w+w_diff, :] = img
                    transformed_images[j] = padded_img
                    
            batch_size = int(len(curated_idx)/len(neurotransmitters)*nb_transformations) # nb of images in per class
            good_datasets = [transformed_images[i:i+batch_size] for i in range(0,len(transformed_images),batch_size)]
            good_datasets = np.array(good_datasets)
            
            transformed_coordinates = np.vstack(transformed_coordinates)
            good_coordinates = [transformed_coordinates[i:i+batch_size] for i in range(0,len(transformed_coordinates),batch_size)]

        else:
            ref_embs_list = []
            for i, index in tqdm(enumerate(indices)):
                dataset_slice = files[i*600:(i+1)*600]
                imgs = [resize_hdf_image(load_image(dataset_slice[k])[0]) for k in index]
                coordinates = [list(map(resize, c)) for c in coords[i]]
                dataset = list(zip(imgs, coordinates))
                class_wise_embs_list = []
                for image, reference in dataset:
                    few_shot.pre_compute_embeddings(
                        image[None,:,:,:],
                        verbose=False,
                        batch_size=1
                    )
                    few_shot.set_reference_vector(get_augmented_coordinates(reference), filter=None)
                    closest_embds = few_shot.get_k_closest_elements(k=k, return_indices=False) # list of vectors
                    class_wise_embs_list.append(torch.mean(closest_embds.cpu(), dim=0)) # list of lists of vectors
                ref_embs_list.append(class_wise_embs_list) # list of lists of lists of vectors

            ref_embs = np.array([np.mean(class_closest_embs, axis=0) for class_closest_embs in ref_embs_list])
            
            torch.save(ref_embs, os.path.join(embeddings_path, f'{model_size}_mean_ref_{resize_size}_Aug={data_aug}_k={k}'))
            
            return ref_embs
        
        
def embedding_generator(batch_size=50):
        
        files, _ = zip(*get_fnames()) 
        images = np.array([resize_hdf_image(load_image(file)[0]) for file in tqdm(files, desc='Loading images')])
        
        for i in range(0,len(images), batch_size):
                batch = images[i:i+batch_size]
                few_shot.pre_compute_embeddings(
                        batch,
                        batch_size=1, 
                        verbose=False
                        )
                torch.cuda.empty_cache()
                gc.collect()
                del batch
                yield few_shot.get_embeddings(reshape=False).cpu()
                
def compute_embeddings(batch_size=50):
        g = embedding_generator()        

        embeddings_list = []

        length = len(images)//batch_size
        for _ in tqdm(range(length)): embeddings_list.append(next(g))
        new_embeddings = torch.cat(embeddings_list)
        torch.save(embeddings_list, os.path.join(embeddings_path, f'{model_size}_dataset_embs_{resize_size}.pt'))
        return embeddings_list
    
    
if __name__ == '__main__':
    compute_embeddings()