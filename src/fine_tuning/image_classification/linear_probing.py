import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os 

from setup import model, device, model_weights_path

from fine_tuning.adaptor.AdaptFormer import augmented_model
from fine_tuning.neuro_classification.Neuro_Classification_Head import head

from .dataset import training_loader, test_loader




ft_loss_fn = nn.CrossEntropyLoss()

block_names = ['DINOv2', 'MLP Head']

def training_block(index, nb_epochs):
    
    if index == 1:

        for param in augmented_model.parameters():
            param.requires_grad = False
        
        for param in augmented_model[1].parameters():
            param.requires_grad = True 


        ft_optimizer = torch.optim.Adam(augmented_model[1].parameters(), lr=3e-4)
        
    elif index == 0:
        
        trainable_params = []
        
        for param in augmented_model.parameters():
            param.requires_grad = False 
        
        for k in range(len(list(augmented_model[0].blocks))):
        
            block = augmented_model[0].blocks[k].mlp
        
            for param in block.down_proj.parameters(): 
                param.requires_grad = True 
                trainable_params.append(param)
            
            for param in block.up_proj.parameters():
                param.requires_grad = True 
                trainable_params.append(param)
            
        ft_optimizer = torch.optim.Adam(trainable_params, lr=3e-4)
        
    augmented_model.to(device)

    loss_list = []
    #prediction_list = []
    accuracy_list = []

    for epoch in tqdm(range(nb_epochs), desc=f'{block_names[index]}'):
        augmented_model.train()
        epoch_loss_list = []

        for image, one_hot in tqdm(ft_training_loader, desc='Training'):
            image = image.to(torch.float32).to(device)
            gt = one_hot.to(torch.float32).to(device)
            
            output = augmented_model(image)
            loss = ft_loss_fn(output, gt)

            loss.backward()
            ft_optimizer.step()
            ft_optimizer.zero_grad()

            epoch_loss_list.append(loss.item())

        loss_list.append(epoch_loss_list)

        # Evaluation
        augmented_model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for image, one_hot in tqdm(ft_test_loader, desc='Test'):
                image = image.to(torch.float32).to(device)
                outputs = augmented_model(image)
                
                for output, target in zip(outputs, one_hot):
                    pred = torch.argmax(output).item()
                    true = torch.argmax(target).item()
                    #prediction_list.append([pred, true])
                    if pred == true:
                        correct += 1
                    total += 1

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)

    for param in augmented_model.parameters():
        param.requires_grad = False 
    
    #torch.save(.state_dict(), os.path.join(model_weights_path, 'Image_Classification_Head_small_first_round.pt'))
    return np.mean(loss_list), np.mean(accuracy_list) # prediction_list


def fine_tuning(nb_iterations, nb_epochs_per_iteration):
    index_list = []
    for _ in range(nb_iterations):
        index_list.extend([1,0])
        
    general_loss_list, general_accuracy_list = [[],[]], [[], []] # first is dino, second is mlp head

    i = 1
    for index in tqdm(index_list, desc='Iterations'):
        loss, accuracy = training_block(index, nb_epochs_per_iteration)
        print(f'Block trained: {block_names[index]} | Iteration: {i}/{nb_iterations} | Train loss: {loss:.3f} | Accuracy: {accuracy:.2f}%')
        i += index
        general_loss_list[index].append(loss)
        general_accuracy_list[index].append(accuracy)
        
    return general_loss_list, general_accuracy_list 