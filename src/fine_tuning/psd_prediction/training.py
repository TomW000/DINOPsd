import torch 
from torch import nn
from tqdm import tqdm
import numpy as np

from .head import detection_head, Psd_Pred_MLP_Head
from .dataset import cross_validation_datasets_generator
from src.fine_tuning.display_results import training_curve, confusion_matrix

from src.setup import device, feat_dim

# the training/testing loop should train on the cross-validated datasets for one epoch, pick the worst case and train for a few more epochs on it.
# It should then return a trained MLP Head that can later be used for semantic segmentation 

# TODO: Add negative anchors
# TODO: plot similarity histogram
# TODO: how about splitting the dataset wrt euclidian distance to any reference vectors

def detection_head_training(nb_epochs, detection_head, train_set, test_set, return_statistics):

    """
    Train and evaluate a detection head using the provided training and testing datasets.

    Args:
        nb_epochs (int): Number of epochs to train the detection head.
        detection_head (nn.Module): The neural network model representing the detection head.
        train_set (DataLoader): DataLoader for the training dataset, providing batches of embeddings and one-hot labels.
        test_set (DataLoader): DataLoader for the testing dataset, providing batches of embeddings and one-hot labels.
        return_statistics (bool): If True, return detailed statistics (loss list, prediction list, and test accuracies), 
                                  otherwise return the final test accuracy and the model's state dictionary.

    Returns:
        Union[Tuple[List[float], List[List[int]], List[float]], Tuple[float, Dict]]: Depending on `return_statistics`, 
        either returns:
            - A tuple containing the list of training losses, prediction pairs, and test accuracies over epochs, or
            - A tuple with the final test accuracy and the state dictionary of the trained detection head.
    """

    loss_list = []
    prediction_list = []
    test_accuracies = []
    
    optimizer = torch.optim.Adam(detection_head.parameters(), lr=3e-4)
    loss_fn = nn.BCELoss()

    for _ in tqdm(range(nb_epochs), desc='Epochs'):
        detection_head.train()
        epoch_loss_list = []

        for embeddings, one_hots in tqdm(train_set, desc='Training', leave=False):
            embeddings = embeddings.to(device)
            one_hots = one_hots.to(device).float()  # Make sure targets are float for BCELoss

            optimizer.zero_grad()
            outputs = detection_head(embeddings).float()  # Ensure float32

            loss = loss_fn(outputs, one_hots)
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())

        loss_list.append(np.mean(epoch_loss_list))

        # Evaluation phase
        detection_head.eval()
        with torch.no_grad():
            score = 0
            total = 0
            for embeddings, one_hots in tqdm(test_set, desc='Testing', leave=False):
                embeddings = embeddings.to(device)
                one_hots = one_hots.to(device)

                outputs = detection_head(embeddings)

                for output, gt in zip(outputs, one_hots):
                    predicted_idx = torch.argmax(output).item()
                    true_idx = torch.argmax(gt).item()
                    prediction_list.append([predicted_idx, true_idx])

                    if predicted_idx == true_idx:
                        score += 1
                    total += 1

            batch_score = 100 * score / total if total > 0 else 0
            test_accuracies.append(batch_score)

        detection_head.train()

    if return_statistics:
        return loss_list, prediction_list, test_accuracies, detection_head.state_dict()
            
    else:
        return test_accuracies[-1]

if __name__ == '__main__':

    untrained_head = detection_head

    dataset_generator = cross_validation_datasets_generator(test_proportion=0.2)

    latest_test_accuracy = 100.0
    most_challenging_dataset = None
    for n, (train_set, test_set) in tqdm(enumerate(dataset_generator)):
        
        detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
        detection_head.to(device)
        test_accuracy = detection_head_training(nb_epochs=1, 
                                                detection_head=detection_head,
                                                train_set=train_set, 
                                                test_set=test_set, 
                                                return_statistics=False) # type: ignore
        print(test_accuracy)
        if test_accuracy < latest_test_accuracy:
            most_challenging_dataset = (train_set, test_set)
        
        latest_test_accuracy = test_accuracy


    train_set, test_set = most_challenging_dataset
    detection_head = untrained_head
    nb_epochs = 10
    loss_list, prediction_list, test_accuracies, head_weights = detection_head_training(nb_epochs=nb_epochs, # type: ignore
                                                                        detection_head=detection_head, 
                                                                        train_set=train_set, 
                                                                        test_set=test_set, 
                                                                        return_statistics=True)

    training_curve(nb_epochs, loss_list, test_accuracies)
    confusion_matrix(data_type='psd',
                     prediction_list=prediction_list, 
                     nb_epochs=nb_epochs,
                     split='test')
