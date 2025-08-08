import torch 
from torch import nn
from tqdm import tqdm
import numpy as np

from .Neuro_Classification_Head import head
from .Neuro_Classification_dataset import training_loader, test_loader
from src.setup import device

optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
loss_fn = nn.BCELoss()


def head_training(epochs):
    head.train()
    loss_list = []
    prediction_list = []
    test_accuracies = []
    for _ in tqdm(range(epochs), desc=f'Epoch:'):
        epoch_loss_list = []
        proportion_list = []
        for embeddings, one_hot_gts in tqdm(training_loader, desc='Training', leave=False):
            embeddings = embeddings.to(device)
            output = head(embeddings).to(torch.float64)
            
            gt = one_hot_gts
            gt = gt.to(device)
            loss=0
            for out, true in zip(output,gt):
                loss += loss_fn(out,true)
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss_list.append(loss.detach().cpu().numpy())
            
            proportion_list.append(one_hot_gts)
                
        loss_list.append(np.mean(epoch_loss_list))

        head.eval()
        with torch.no_grad():
            score = 0
            total = 0
            for embeddings, one_hot_gts in tqdm(test_loader, desc='Testing', leave=False):
                embeddings = embeddings.to(device)
                outputs = head(embeddings) # shape (batch_size, nb_classes)
                
                for output, one_hot_gt in zip(outputs, one_hot_gts):
                    predicted_idx = torch.argmax(output).item()
                    true_idx = torch.argmax(one_hot_gt).item()
                    prediction_list.append([predicted_idx, true_idx])
                    
                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                batch_score = 100*score/total
            test_accuracies.append(batch_score)

    return loss_list, proportion_list, prediction_list, test_accuracies