from setup import neurotransmitters, model, device, feat_dim, resize_size, dataset_path, curated_idx
from setup import tqdm, torch, np, os, h5py, sns, plt
from setup import cosine_similarity, euclidean_distances
from perso_utils import get_fnames, load_image, get_latents
from personalized_pipeline.DINOPsd import DinoSim_pipeline, diplay_features
from napari_dinosim.utils import get_img_processing_f


if __name__=='__main__':  # Fixed: Added double underscores
    
    # Create an instance of the pipeline (not just assigning the class)
    few_shot = DinoSim_pipeline(model,
                                model.patch_size,
                                device,
                                get_img_processing_f(resize_size),
                                feat_dim, 
                                dino_image_size=resize_size
                                )

    k = 10  
    d = 0.5

    files, labs = zip(*get_fnames()) # returns list(zip(np.concatenate(files), np.concatenate(labels)))
    good_files = [files[idx] for idx in curated_idx] # len = 30
    datasets = [good_files[i:i+10] for i in range(0, len(curated_idx), 10)] # list of lists of files
    good_labels = [labs[idx] for idx in curated_idx]
    _labels = [good_labels[i:i+10] for i in range(0, len(curated_idx), 10)] # list of lists of labels
    
    latent_list, label_list = [], []

    for dataset, batch_label in tqdm(zip(datasets, _labels), desc='Iterating through neurotransmitters'):
    #for d, l in tqdm(zip(good_files, good_labels), desc='Iterating through neurotransmitters'):
        # Load images and prepare data
        images = np.array([load_image(file)[0] for file in dataset]).transpose(0,2,3,1)  # Convert to numpy array
        coordinates = [(0, load_image(file)[1], load_image(file)[2]) for file in dataset]

        # Pre-compute embeddings
        few_shot.pre_compute_embeddings(
            images,  # Pass numpy array of images
            overlap=(0.5, 0.5),
            padding=(0, 0),
            crop_shape=(512, 512, 1),
            verbose=True,
            batch_size=5
        )
        
        # Set reference vectors
        few_shot.set_reference_vector(coordinates, filter=None)
        ref = few_shot.get_refs()
        
        # Get closest elements - using the correct method name
        
        close_embedding = few_shot.get_k_closest_elements(
            k=k
        )
        #k_labels = [l for _ in range(k)]
        k_labels = [batch_label[0] for _ in range(k)]
    
        
        # Convert to numpy for storing
        close_embedding_np = close_embedding.cpu().numpy() if isinstance(close_embedding, torch.Tensor) else close_embedding
        
        latent_list.append(close_embedding_np)
        label_list.append(k_labels)
        
        # Clean up to free memory
        few_shot.delete_precomputed_embeddings()
        few_shot.delete_references()
    
    img = np.array(load_image(files[0])[0])[...,np.newaxis]#.transpose(0,2,3,1)
    
    few_shot.pre_compute_embeddings(
            img,  # Pass numpy array of images
            overlap=(0.5, 0.5),
            padding=(0, 0),
            crop_shape=(512, 512, 1),
            verbose=True,
            batch_size=1
        )
    
    emb = few_shot.get_embs().reshape(-1, feat_dim)
    
    new = ['new' for _ in range(emb.shape[0])]
    
    # Stack all embeddings and labels
    display_latents = np.vstack(latent_list)
    display_labels = np.hstack(label_list)
    
    latents = np.vstack([display_latents, emb])  # Changed from stack to vstack for proper concatenation
    labels = np.hstack([display_labels, new])    # Changed from stack to hstack for proper concatenation
    
    mean_ref = np.vstack([np.mean(list, axis=0) for list in latent_list])
    mean_labs = [neurotransmitter for neurotransmitter in neurotransmitters]
    

# Compute cosine similarity matrix
    similarity_matrix = euclidian_distance(mean_ref, display_latents)

    plt.figure(figsize=(10,10),dpi=300)
    sns.heatmap(similarity_matrix, yticklabels=mean_labs)
    plt.title('Cosine simililarity matrix')
    plt.xlabel('Image patches')
    plt.ylabel('Reference embeddings')

    
    
    
'''
    diplay_features(
        
        latents = np.vstack([display_latents, np.vstack(mean_ref)]),
        labels = np.hstack([display_labels, np.hstack(mean_labs)]),

        include_pca=False,
        pca_nb_components=100,
        clustering=False,
        nb_clusters=6,
        nb_neighbor=20,
        min_dist=0.1,
        nb_components=2,
        metric='cosine'
    )

    mito_paths = ['sylee_neurotrans_cubes_18Feb2025/acetylcholine/acetylcholine_1612-1742_y17627-17757_z10893-11023_1850706912.hdf', 
                  'sylee_neurotrans_cubes_18Feb2025/acetylcholine/acetylcholine_11383-11513_y13916-14046_z12575-12705_1630051981.hdf', 
                  'sylee_neurotrans_cubes_18Feb2025/dopamine/dopamine_14954-15084_y19804-19934_z16506-16636_950229431.hdf', 
                  'sylee_neurotrans_cubes_18Feb2025/dopamine/dopamine_28915-29045_y17216-17346_z13299-13429_759464163.hdf', 
                  'sylee_neurotrans_cubes_18Feb2025/gaba/gaba_6860-6990_y16922-17052_z14280-14410_757556799.hdf',
                  'sylee_neurotrans_cubes_18Feb2025/gaba/gaba_15193-15323_y31507-31637_z17036-17166_425790257.hdf',  
                  'sylee_neurotrans_cubes_18Feb2025/glutamate/glutamate_18611-18741_y31015-31145_z13535-13665_612371421.hdf',  
                  'sylee_neurotrans_cubes_18Feb2025/glutamate/glutamate_19705-19835_y12969-13099_z14794-14924_910783961.hdf', 
                  'sylee_neurotrans_cubes_18Feb2025/octopamine/octopamine_6034-6164_y13729-13859_z27057-27187_5813049966.hdf',
                  'sylee_neurotrans_cubes_18Feb2025/octopamine/octopamine_12369-12499_y19250-19380_z15493-15623_821344462.hdf',
                  'sylee_neurotrans_cubes_18Feb2025/serotonin/serotonin_14261-14391_y26426-26556_z5122-5252_297230760.hdf',
                  'sylee_neurotrans_cubes_18Feb2025/serotonin/serotonin_17928-18058_y33523-33653_z26353-26483_759810119.hdf'
                  ]
    
    _MITO_labels = [['MITO_Acetylcholine' for _ in range(k)], 
                   ['MITO_Acetylcholine' for _ in range(k)], 
                   ['MITO_Dopamine' for _ in range(k)], 
                   ['MITO_Dopamine' for _ in range(k)], 
                   ['MITO_GABA' for _ in range(k)], 
                   ['MITO_GABA' for _ in range(k)], 
                   ['MITO_Glutamate' for _ in range(k)], 
                   ['MITO_Glutamate' for _ in range(k)], 
                   ['MITO_Octapamine' for _ in range(k)], 
                   ['MITO_Octapamine' for _ in range(k)], 
                   ['MITO_Serotonin' for _ in range(k)], 
                   ['MITO_Serotonin' for _ in range(k)]]
    
    MITO_labels = sum(_MITO_labels, [])


    z_idx = [60, 63, 31, 31, 84, 30, 64, 20, 16, 110, 1, 1]

    mito_coordinates = [(0,70,110), (0,22,15), (0,10,35), (0,105,40), (0,105,40), (0,70,50), (0,80,110), (0,10,10), (0,80,95), (0,50,70), (0,70,50), (0,55,70)]


    MITO_latent_list = []

    for mito_file, idx, mito_coord in tqdm(zip(mito_paths, z_idx, mito_coordinates), desc='Iterating through negative examples'):
        mito_path = os.path.join(dataset_path, mito_file)
        with h5py.File(mito_path) as f:
            img = f['volumes/raw'][:][np.newaxis,:,:,idx, np.newaxis]#.transpose(0,2,3,1)

        # Pre-compute embeddings
        few_shot.pre_compute_embeddings(
            img,  # Pass numpy array of images
            overlap=(0.5, 0.5),
            padding=(0, 0),
            crop_shape=(512, 512, 1),
            verbose=True,
            batch_size=1
        )
        # Set reference vectors
        few_shot.set_reference_vector([mito_coord], filter=None)
        
        # Get closest elements - using the correct method name
        close_embedding = few_shot.get_k_closest_elements(
            k=k
        )
        
        close_embedding, _, _ = few_shot.get_d_closest_elements(
            d=0.05,
            verbose=True
        )
        
        # Convert to numpy for storing
        close_embedding_np = close_embedding.cpu().numpy() if isinstance(close_embedding, torch.Tensor) else close_embedding
        
        MITO_latent_list.append(close_embedding_np)
        
        # Clean up to free memory
        few_shot.delete_precomputed_embeddings()
        few_shot.delete_references()
    
    # Stack all embeddings and labels
    MITO_latents = np.vstack(MITO_latent_list)  # Changed from stack to vstack for proper concatenation
    
    lat = np.concatenate((latents, MITO_latents), axis=0)
    lab = [*labels, *MITO_labels]


    data, labs = zip(*get_fnames())
    datasetss = [data[i:i+300] for i in range(0,300,300)]
    lats_list, labs_list = [], []
    for ds in tqdm(zip(datasetss, labs)):
        lats, labs = get_latents(dataset_paths=data, labels=labs, k=5)
        lats_list.append(lats)
        labs_list.append(labs)
    all_latents, all_labels = np.vstack(lats_list), np.hstack(labs_list)
    print('Preparing display 2')
    
    diplay_features(
        all_latents,
        all_labels,
        include_pca=True,
        pca_nb_components=50,
        clustering=False,
        nb_clusters=6,
        nb_neighbor=15,
        min_dist=0.05,
        nb_components=2,
        metric='cosine'
    )
    '''