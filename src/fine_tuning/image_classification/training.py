import torch 
from torch import nn
from tqdm import tqdm
import numpy as np

from src.fine_tuning.neuro_classification.Neuro_Classification_Head import MLP_Head  # Assuming this is an instantiated model
from .dataset import get_dataset
from src.setup import device, feat_dim, model
from src.fine_tuning.display_results import confusion_matrix, training_curve

head = MLP_Head(device=device, feat_dim=feat_dim)

for param in head.parameters():
    param.requires_grad = True
    
for praram in model.parameters():
    param.requires_grad = False

complete_model = nn.Sequential(model, head)

# Use CrossEntropyLoss for multi-class classification
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)


def head_training(epochs, training_set, test_set):
    head.to(device)
    head.train()
    
    loss_list = []
    prediction_list = []
    test_accuracies = []

    for _ in tqdm(range(epochs), desc=f'Epoch:'):
        epoch_loss_list = []

        for images, one_hot_gts in tqdm(training_set, desc='Training', leave=False):
            images = images.to(device).to(torch.float32)
            one_hot_gts = one_hot_gts.to(device)

            # Convert one-hot to class indices
            class_indices = torch.argmax(one_hot_gts, dim=1)

            output = complete_model(images)
            loss = loss_fn(output, class_indices)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_list.append(loss.item())

        loss_list.append(np.mean(epoch_loss_list))

        # Evaluation
        head.eval()
        score = 0
        total = 0

        with torch.no_grad():
            for images, one_hot_gts in tqdm(test_set, desc='Testing', leave=False):
                images = images.to(device).to(torch.float32)
                one_hot_gts = one_hot_gts.to(device)

                outputs = complete_model(images)
                predicted_classes = torch.argmax(outputs, dim=1)
                true_classes = torch.argmax(one_hot_gts, dim=1)

                prediction_list.extend(zip(predicted_classes.cpu().numpy(), true_classes.cpu().numpy()))

                score += (predicted_classes == true_classes).sum().item()
                total += true_classes.size(0)

        accuracy = 100 * score / total
        test_accuracies.append(accuracy)

    return loss_list, prediction_list, test_accuracies


if __name__ == '__main__':
    crop_size = 98
    training_set, test_set = get_dataset(crop_size=crop_size)
    print(f'-Cropping size: {crop_size}')
    
    nb_epochs = 10
    loss_list, prediction_list, test_accuracies = head_training(
        epochs=nb_epochs,
        training_set=training_set,
        test_set=test_set
    )
    
    training_curve(nb_epochs=nb_epochs, loss_list=loss_list, test_accuracies=test_accuracies)
    confusion_matrix(data_type='neuro', prediction_list=prediction_list, nb_epochs=nb_epochs, split='test')
