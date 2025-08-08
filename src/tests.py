from src.fine_tuning.neuro_classification.Neuro_Classification_Head import head
from src.fine_tuning.neuro_classification.Neuro_Classification_training import head_training

from src.fine_tuning.display_results import confusion_matrix, training_curve, class_proportions

if __name__ == '__main__':

    nb_epochs = 1
    loss_list, proportion_list, prediction_list, test_accuracies = head_training(nb_epochs)

    #class_proprtions(prediction_list)

    split = '80/20'
    confusion_matrix(data_type='neuro', prediction_list=prediction_list, nb_epochs=nb_epochs, split=split)
    #training_curve(nb_epochs, loss_list, test_accuracies)