import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import umap

from src.setup import model, device, model_weights_path

from src.fine_tuning.adaptor.AdaptFormer import augmented_model

from .training import train_loader, test_loader


def training_block(model,
                   index, 
                   nb_epochs, 
                   train_set, 
                   test_set, 
                   return_statistics, 
                   use_umap):

    loss_fn = nn.BCELoss()

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
    test_accuracies = []

    for epoch in tqdm(range(nb_epochs), desc='--> Epochs'):
        model.train()
        epoch_loss_list = []

        for embeddings, ground_truths in tqdm(train_set, desc='-> Training', leave=False):
            embeddings = embeddings.to(device)
            ground_truths = ground_truths.to(device).float().unsqueeze(1)  # Make sure targets are float for BCELoss

            optimizer.zero_grad()
            outputs = model(embeddings).float() # Ensure float32

            loss = loss_fn(outputs, ground_truths)
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())

        loss_list.append(np.mean(epoch_loss_list))

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            score = 0
            total = 0
            
            #UMAP
            if use_umap and epoch == 0:
                UMAP_LIST, UMAP_LABELS = [], []
                UMAP_EMBEDDINGS = []
                
            for embeddings, ground_truths in tqdm(test_set, desc='-> Testing', leave=False):
                
                if use_umap and epoch == 0:    
                    #UMAP:
                    UMAP_LIST.extend(embeddings)
                    UMAP_LABELS.extend(ground_truths)
                    
                
                embeddings = embeddings.to(device)
                ground_truths = ground_truths.to(device)
                

                outputs = model(embeddings)

                for output, gt in zip(outputs, ground_truths):
                    predicted_idx = torch.round(output)
                    true_idx = gt
                    assert type(predicted_idx) == type(true_idx), "predicted_idx and true_idx must be of the same type"

                    prediction_list.append([int(predicted_idx), int(true_idx)])

                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                    
            batch_score = 100 * score / total if total > 0 else 0
            test_accuracies.append(batch_score)


    #UMAP
    if use_umap:
        print('Running UMAP')
        UMAP_EMBEDDINGS = reducer.fit_transform(UMAP_LIST)

        plt.clf()
        unique_labels = np.unique(UMAP_LABELS)
        
        colors = plt.cm.bwr(np.linspace(0,1,len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = UMAP_LABELS == label
            if label == 0:
                plot_label = 'Rest'
            else:
                plot_label = 'PSD'
        plt.scatter(UMAP_EMBEDDINGS[mask, 0], UMAP_EMBEDDINGS[mask, 1], c=[colors[i]], s=0.5, label=plot_label, marker='o')
        plt.legend()
        plt.title(f'UMAP - accuracy: {batch_score}')
        plt.show()
        #plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', f'UMAP_{batch_score:.2f}.png'))


    if return_statistics:
        return loss_list, prediction_list, test_accuracies, model.state_dict()

    else:
        return loss_list[-1], test_accuracies[-1]


def fine_tuning(model,
                train_set,
                test_set,
                nb_iterations, 
                nb_epochs_per_iteration):
    block_names = ['DINOv2', 'MLP Head']
    index_list = []
    for _ in range(nb_iterations):
        index_list.extend([1,0])
        
    general_loss_list, general_accuracy_list = [[],[]], [[], []] # first is dino, second is mlp head

    i = 1
    for index in tqdm(index_list, desc='Iterations'):

        loss, accuracy = training_block(model=model,
                                        index=index, 
                                        nb_epochs=nb_epochs_per_iteration, 
                                        train_set=train_set, 
                                        test_set=test_set, 
                                        return_statistics=False, 
                                        use_umap=True)

        print(f'Block trained: {block_names[index]} | Iteration: {i}/{nb_iterations} | Train loss: {loss:.3f} | Accuracy: {accuracy:.2f}%')
        i += index
        general_loss_list[index].append(loss)
        general_accuracy_list[index].append(accuracy)
        
    return general_loss_list, general_accuracy_list 

if __name__ == '__main__':
    general_loss_list, general_accuracy_list = fine_tuning(model=augmented_model,
                train_set=train_loader,
                test_set=test_loader,
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