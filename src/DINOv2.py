from setup import model, Trans, umap, torch, tqdm, np, px, PCA, os, glob, curated_idx
from perso_utils import get_processed_image, get_fnames


def get_embeddings(batch_size=1):

    transform = Trans.Compose([
        Trans.ToTensor(),
        Trans.Resize((224, 224)),
        Trans.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

    files, labs = zip(*get_fnames()) # returns list(zip(np.concatenate(files), np.concatenate(labels)))
    fnames = [files[idx] for idx in curated_idx] # len = 60
    labels = [labs[idx] for idx in curated_idx]
    latent, img_batch = [], []
    
    for file in tqdm(fnames, desc='Computing embeddings'):
        try:
            # Get and process the image
            img_c = get_processed_image(file)[3][0,...,0]
            img_rgb = np.stack([img_c, img_c, img_c], axis=2)
            img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
            img_batch.append(img_tensor)
            
            # Process batch when we reach batch_size
            if len(img_batch) == batch_size:
                batch_tensor = torch.cat(img_batch, dim=0)
                with torch.no_grad():
                    features = model(batch_tensor)
                    latent.append(features.cpu().numpy())
                img_batch = []  # Clear the batch
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # Process any remaining images in the last batch
    if img_batch:
        batch_tensor = torch.cat(img_batch, dim=0)
        with torch.no_grad():
            features = model(batch_tensor)
            latent.append(features.cpu().numpy())

    # Concatenate all batches of features
    if not latent:
        return None, labels
    return np.stack(latent).squeeze(), labels

#----------------------------------------------------------------------------------------------------------------------------------------------

def diplay_features(embeddings,
                    labels,
                    include_pca=True,
                    pca_nb_components=100,
                    nb_neighbor=5,
                    min_dist=0.01,
                    nb_components=2,
                    metric='correlation'):

    if len(embeddings):

        if include_pca:
            pca = PCA(n_components=pca_nb_components)
            features = pca.fit_transform(embeddings)
        else:
            features = embeddings

        reducer = umap.UMAP(
            n_neighbors=nb_neighbor,
            min_dist=min_dist,
            n_components=nb_components,
            metric=metric
            )
        embedding = reducer.fit_transform(features)

        fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color = labels,
            title = f'PCA={include_pca} ({pca_nb_components}) - UMAP (n_neighbors={nb_neighbor}, min_dist={min_dist}, n_components={nb_components}, metric={metric})',
            width=1500,
            height=1000
        )
        fig.show()
    else:
        print("No features were extracted!")


if __name__=='__main__':
    '''
    batch_size = int(input('Batch size: '))
    include_pca = bool(input('Include PCA? (Boolean): '))
    if include_pca:
        pca_nb_components = int(input('Number of PCA components: '))
    nb_neighbor = int(input('Number of neighbors: '))
    min_dist = float(input('Minimum distance: '))
    nb_components = int(input('Number of components: '))
    metric = input('Metric: ')
    embeddings, labels = get_embeddings(batch_size)
    diplay_features(embeddings, labels, include_pca, pca_nb_components, nb_neighbor, min_dist, nb_components, metric) 
    '''
    embeddings, labels = get_embeddings(batch_size=1)
    diplay_features(embeddings,
                    labels,
                    include_pca=True,
                    pca_nb_components=50,
                    nb_neighbor=15,
                    min_dist=0.1,
                    nb_components=2,
                    metric='cosine')