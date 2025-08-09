import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from .AdaptFormer import augmented_model, device 
from .AdaptFormer_dataset import ft_training_loader, ft_test_loader 


ft_optimizer = torch.optim.Adam(augmented_model.parameters(), lr=3e-4)
augmented_model.to(device)
ft_loss_fn = nn.BCELoss()

def adaptor_training(epochs):
    augmented_model.train()
    loss_list = []
    prediction_list = []
    test_accuracies = []
    for _ in tqdm(range(epochs), desc=f'Epoch:'):
        epoch_loss_list = []
        for images, one_hot_gts in tqdm(ft_training_loader, desc='Training', leave=False):
            images = images.to(torch.float32).to(device)
            output = augmented_model(images).to(torch.float64)
            
            gt = one_hot_gts
            gt = gt.to(device)
            loss=0
            for out, true in zip(output,gt):
                loss += ft_loss_fn(out,true)
                
            loss.backward()
            ft_optimizer.step()
            ft_optimizer.zero_grad()
            
            epoch_loss_list.append(loss.detach().cpu().numpy())

        loss_list.append(np.mean(epoch_loss_list))

        augmented_model.eval()
        with torch.no_grad():
            score = 0
            total = 0
            for images, one_hot_gts in tqdm(ft_test_loader, desc='Testing', leave=False):
                
                images = images.to(torch.float32).to(device)
                outputs = augmented_model(images) # shape (batch_size, nb_classes)
                
                for output, one_hot_gt in zip(outputs, one_hot_gts):
                    predicted_idx = torch.argmax(output).item()
                    true_idx = torch.argmax(one_hot_gt).item()
                    prediction_list.append([predicted_idx, true_idx])
                    
                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                batch_score = 100*score/total
            test_accuracies.append(batch_score)

    return loss_list, prediction_list, test_accuracies