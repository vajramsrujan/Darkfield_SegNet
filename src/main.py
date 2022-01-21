# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:22:06 2022

@author: vajra
"""



import torch
import numpy as np
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation

from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn     # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar
from classes.SegNetDataSet import SegNetDataSet
from classes.SegNet import SegNet
from functions import *

# ============================================================================= # 

# Empty cuda cache memory
torch.cuda.empty_cache()

# Compose transformations for data
data_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.ToTensor(),  
    transforms.Normalize( mean = [0.1600, 0.1959, 0.2559], 
                          std=[0.2209, 0.2456, 0.2530] )
    ])   

# Separate transforms for targets
target_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    ])   

# Hyperparameters
in_channels = 3
learning_rate = 0.01
batch_size = 16
num_epochs = 35
num_classes = 3
LOAD_MODEL = False

# Load custom dataset
dataset = SegNetDataSet(r'C:\Users\vajra\Documents\GitHub\Darkfield_SegNet\src\archive', 
                        data_transforms=data_transforms, target_transforms=target_transforms)

# Produce test and train sets
train_set, test_set = torch.utils.data.random_split(dataset, [329, 37]) # 90% 10% split between train and test 

# Create train and test loaders
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if not LOAD_MODEL: 
    # Initialize network
    model = SegNet(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # Set the weighting for each class used in the loss
    normalized_weights = [0.1664200524791033, 0.8427401371639913, 0.9908398103569054]
    normalized_weights = torch.FloatTensor(normalized_weights).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(normalized_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train Network
    losses = np.zeros((1,num_epochs)).flatten()
    avg_training_acc = np.zeros((2,num_epochs))
    avg_testing_acc = np.zeros((2,num_epochs))
    
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
    
            # Forward pass
            scores = model(data)
            
            # Compute loss
            loss = criterion(scores, targets)
            losses[epoch] = loss.item()
            
            # Backward propagation 
            optimizer.zero_grad() # Make sure to reset gradients
            loss.backward()       # Backprop
    
            # Gradient descent step with optimizer
            optimizer.step()
      
    # Upon training completion, save model as .tar file
    state = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict()}
    save_model_at_checkpoint(state, epoch)
    
else:
    # If LOAD_MODEL = True, laod the saved .tar model paramters
    model = SegNet(in_channels=in_channels, num_classes=num_classes).to(device)
    model, optimizer = load_model("saved_model/model_at_epoch_" + str(num_epochs) + ".pth.tar", model, learning_rate)
    
# ============================================================================= # 

# Computes the segmentation accuracies per class for each image in the train and test sets. 
# Also returns a list of predictions and associated truth masks
training_accuracies, training_predictions = check_accuracy(train_loader, model, num_classes)
testing_accuracies, test_predictions  = check_accuracy(test_loader, model, num_classes)

# Prints segmentation accuracies of training and test sets for RBC class (class 1) and bacteria class (class 2)
print("\n")
for i in range(1, len(training_accuracies)):
    print("training accuracy for class " + str(i) + " is:")
    print(training_accuracies[i].mean())
    print("test accuracy for class " + str(i) + " is:")
    print(testing_accuracies[i].mean()) 
    
# You can view a prediction and its corresponding truth mask using show_overlay()
# Ex: will show training prediction for for image 25 in the dataloader and the truth mask 
show_overlay(training_predictions, 25)
