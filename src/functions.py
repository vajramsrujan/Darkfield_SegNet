# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:17:40 2022

@author: vajra
"""

import torch
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
# ============================================================================= # 

def load_model(filename, model, learning_rate): 
    
    '''
        Will load pre-saved model and optimizer
    '''
    
    # Load optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load model parameters using fil.tar filename
    checkpoint = torch.load(filename, map_location=torch.device('cpu')) # map_location sets mapping to cpu
    # Load the states of the model and optimizer
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    
    return model, optimizer
# ============================================================================= # 

def save_model_at_checkpoint(state, epoch):
    
    '''
        Saves model at specific epoch number as .tar file
    '''
    
    # Set filename for saving
    filename = "model_at_epoch_" + str(epoch+1) + ".pth.tar"
    # Save
    torch.save(state, filename)
    
# ============================================================================= # 
# # Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, num_classes):
    
    ''' 
        Computes segmentation accuracy per class for each image.
        Returns list of accuracies per image per class, and list of 
        prediction masks and associated truth masks
    '''
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model to evaluation mode. Prevents things like dropout from occuring
    # which are only meant to b e used during the training phase 
    model.eval()
    
    # Initialize empty lists which will hold the segmentation accuracy per image per class,
    # and predictions 
    class_accuracies = []
    for i in range(num_classes):
        # In each iteration, add an empty list to the main list
        class_accuracies.append([])
    
    predictions = []
    
    # torch.no_grad ensures we are not computing gradients
    with torch.no_grad():
        for index, (data, target) in enumerate(loader):
            
            print("checking batch " + str(index))
            # Send data to device
            data = data.to(device=device)
            # Compute predictions for batch
            scores = model(data)
            
            # Grab each prediction image in the mini_batch
            for i in range(scores.shape[0]):
                score = scores[i, :, :, :]
                
                # Collapse each multi-channel prediction image to a single numpy matrix
                # (in this case, 3 classes = 3 channels.)
                prediction = torch.argmax(score.squeeze(), dim=0).cpu().detach().numpy()
                # Grab associated truth mask and convert to numpy matrix
                true_label = target[i, :, :].numpy()
                
                # Append the prediction mask and associated truth mask to list
                predictions.append((prediction, true_label))
                
                # For each class
                for n in range(num_classes):
                
                    # Compute a binary map of pixel positions per class for the prediction and truth masks
                    prediction_class_pos = (prediction == n)
                    true_label_class_pos = (true_label == n)
                    
                    # Logical AND operator to figure out overlapping (i.e correct) prediction area
                    matching = prediction_class_pos & true_label_class_pos
                    # Logical XOR operation to figure out extraneous (incorrect) prediction area
                    unmatching = prediction_class_pos ^ true_label_class_pos # xor
                    
                    # If an instance of the class in question exists in the truth label
                    if true_label_class_pos.sum() != 0:
                        # Compute accuracy by subtracting proportion of 
                        # incorrectly predicted  area from proportion of correctly predicted area
                        accuracy = (matching.sum() / true_label_class_pos.sum()) - (unmatching.sum() / len(unmatching.flatten()))
                    
                    else:
                        # Otherwise only penalize any incorrectly predicted area
                        accuracy = 1 - ( (unmatching.sum() / len(unmatching.flatten())) )
                    
                    class_accuracies[n].append(accuracy*100)
        
    # Revert model mode to training
    model.train()
    return np.array(class_accuracies), predictions

# ============================================================================= # 

def show_overlay(prediction_list, index): 

    '''
        Plots the prediction mask and the associated turh mask 
        for a specific image in the passed loader
    '''
    
    # Plot prediction mask
    plt.figure()
    plt.title("Prediction")
    plt.imshow(prediction_list[index][0])
    # Plot truth mask
    plt.figure()
    plt.title("True mask")
    plt.imshow(prediction_list[index][1])
        
    return

# ============================================================================= # 