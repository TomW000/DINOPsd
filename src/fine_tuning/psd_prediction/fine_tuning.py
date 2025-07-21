import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import f1_score

from src.setup import model, device, model_weights_path, resize_size, feat_dim

from src.fine_tuning.adaptor.AdaptFormer import augmented_model
from .sliding_window_dataset import get_dataset

def training_block(model,
                   index, 
                   nb_epochs, 
                   train_set, 
                   test_set, 
                   return_statistics, 
                   use_umap):

    if index == 1:

        for param in model.parameters():
            param.requires_grad = False
        
        for param in model[1].parameters():
            param.requires_grad = True 


        optimizer = torch.optim.Adam(model[1].parameters(), lr=3e-4)
        
    elif index == 0:
        
        trainable_params = []
        
        for param in model.parameters():
            param.requires_grad = False 
        
        for k in range(len(list(model[0].blocks))):
        
            block = model[0].blocks[k].mlp
        
            for param in block.down_proj.parameters(): 
                param.requires_grad = True 
                trainable_params.append(param)
            
            for param in block.up_proj.parameters():
                param.requires_grad = True 
                trainable_params.append(param)
            
        optimizer = torch.optim.Adam(trainable_params, lr=3e-4)
        
    model.to(device)

    #UMAP
    if use_umap:
        reducer = umap.UMAP(random_state=42)

    loss_list = []
    prediction_list = []
    test_loss_list = []

    patch_size = model[0].patch_size
    def loss_fn(y_true, y_pred): f1_score(y_true, y_pred)

    for epoch in tqdm(range(nb_epochs), desc='--> Epochs'):
        model.train()
        epoch_loss_list = []

        for patches, ground_truth in tqdm(train_set, desc='-> Training', leave=False):
            ground_truth = ground_truth.to(device)
            prediction = torch.zeros(ground_truth.shape)
            H_patch, W_patch = ground_truth.shape

            optimizer.zero_grad()
            for n, patch in enumerate(patches):
                patch = patch.to(device)
                nb_patches_y, nb_patches_x = patch.shape[1:]
                embeddings = model[0].forward_features(patch)["x_norm_patchtokens"].reshape(nb_patches_y, nb_patches_x, feat_dim)
                central_embedding = embeddings[nb_patches_y//2, nb_patches_x//2]
                central_embedding = central_embedding.unsqueeze(0)
                output = model[1](central_embedding)
                y_coord = n // H_patch
                x_coord = n % W_patch
                prediction[y_coord: y_coord + patch_size, x_coord: x_coord + patch_size] = torch.zeros(patch_size, patch_size) if round(output) == 0 else torch.ones(patch_size, patch_size)
                
            loss = loss_fn(ground_truth, prediction)
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss)

        loss_list.append(np.mean(epoch_loss_list))

        # Evaluation phase
        model.eval()
        with torch.no_grad():

            #UMAP
            if use_umap and epoch == 0:
                UMAP_LIST, UMAP_PREDICTED_LABELS = [], []
                UMAP_EMBEDDINGS = []

            for patches, ground_truth in tqdm(test_set, desc='-> Testing', leave=False):
                ground_truth = ground_truth.to(device)
                prediction = torch.zeros(ground_truth.shape)
                H_patch, W_patch = ground_truth.shape

                optimizer.zero_grad()
                for n, patch in enumerate(patches):
                    patch = patch.to(device)
                    nb_patches_y, nb_patches_x = patch.shape[1:]
                    embeddings = model[0].forward_features(patch)["x_norm_patchtokens"].reshape(nb_patches_y, nb_patches_x, feat_dim)
                    central_embedding = embeddings[nb_patches_y//2, nb_patches_x//2]
                    central_embedding = central_embedding.unsqueeze(0)
                    output = model[1](central_embedding)
                    #UMAP:
                    if use_umap and epoch == 0:    
                        UMAP_LIST.extend(central_embedding.cpu().numpy())
                        UMAP_PREDICTED_LABELS.extend(output.cpu().numpy())
                    y_coord = n // H_patch
                    x_coord = n % W_patch
                    prediction[y_coord: y_coord + patch_size, x_coord: x_coord + patch_size] = torch.zeros(patch_size, patch_size) if round(output) == 0 else torch.ones(patch_size, patch_size)
                    
                loss = loss_fn(ground_truth, prediction)
            test_loss_list.append(loss)

    #UMAP
    if use_umap:
        print('Running UMAP')
        UMAP_EMBEDDINGS = reducer.fit_transform(UMAP_LIST)

        plt.clf()
        unique_labels = np.unique(UMAP_PREDICTED_LABELS)
        
        colors = plt.cm.bwr(np.linspace(0,1,len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = UMAP_PREDICTED_LABELS == label
            if label == 0:
                plot_label = 'Rest'
            else:
                plot_label = 'PSD'
        plt.scatter(UMAP_EMBEDDINGS[mask, 0], UMAP_EMBEDDINGS[mask, 1], c=[colors[i]], s=0.5, label=plot_label, marker='o')
        plt.legend()
        plt.title(f'UMAP - accuracy: {loss}')
        plt.show()
        #plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', f'UMAP_{batch_score:.2f}.png'))


    if return_statistics:
        return loss_list, prediction_list, test_loss_list, model.state_dict()

    else:
        return loss_list[-1], test_loss_list[-1]


def fine_tuning(model,
                nb_iterations, 
                nb_epochs_per_iteration):
    block_names = ['DINOv2', 'MLP Head']
    index_list = []
    for _ in range(nb_iterations):
        index_list.extend([1,0])
        
    general_loss_list, general_accuracy_list = [[],[]], [[], []] # first is dino, second is mlp head

    train_set, test_set = get_dataset(nb_best_patches=1,
                                      resize_size=resize_size, 
                                      padding_size=5, 
                                      test_proportion=0.2,
                                      batch_size=1)


    i = 1
    for index in tqdm(index_list, desc='Iterations'):

        loss, test_loss = training_block(model=model,
                                        index=index, 
                                        nb_epochs=nb_epochs_per_iteration, 
                                        train_set=train_set, 
                                        test_set=test_set, 
                                        return_statistics=False, 
                                        use_umap=True)

        print(f'Block trained: {block_names[index]} | Iteration: {i}/{nb_iterations} | Train loss: {loss:.3f} | Accuracy: {accuracy:.2f}%')
        i += index
        general_loss_list[index].append(loss)
        general_accuracy_list[index].append(test_loss)
        
    return general_loss_list, general_accuracy_list 

if __name__ == '__main__':
    general_loss_list, general_accuracy_list = fine_tuning(model=augmented_model,
                                                           nb_iterations=2, 
                                                           nb_epochs_per_iteration=40)

    x = [i for i in range(len(general_loss_list[0]))]
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=150)
    ax2 = ax1.twinx()

    # Plot curves
    lns1 = ax1.plot(x, general_loss_list[0], label='DINO Train Loss', color='blue')
    lns1_1 = ax1.plot(x, general_loss_list[1], label='MLP Train Loss', color='cyan')
    ax1.set_ylim(0, max(sum(general_loss_list), []) * 1.05)
    ax1.set_ylabel('Train Loss')

    lns2 = ax2.plot(x, general_accuracy_list[0], label='DINO Test Accuracy', color='red')
    lns2_2 = ax2.plot(x, general_accuracy_list[1], label='MLP Test Accuracy', color='orange')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Test Accuracy')

    # Combine legends
    lns = lns1 + lns1_1 + lns2 + lns2_2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc=0)

    ax1.set_xlabel('Iterations')
    plt.title('Training Curve')
    plt.tight_layout()
    plt.show()