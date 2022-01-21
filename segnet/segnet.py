#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:28:23 2022

@author: srujanvajram
"""

# Imports
import numpy as np

import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from SegNetDataSet import SegNetDataSet
from PIL import Image
from matplotlib import pyplot as plt

# ============================================================================= # 
#  SegNet
class SegNet(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        
        # -------------------------# 
        # Encoder block 1
        self.encoder_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn2 = nn.BatchNorm2d(64)
        
        self.encoder_mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # -------------------------# 
        # Encoder block 2
        self.encoder_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn4 = nn.BatchNorm2d(128)
        
        self.encoder_mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # -------------------------# 
        # Encoder block 3
        self.encoder_conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn5 = nn.BatchNorm2d(256)
        self.encoder_conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_bn6 = nn.BatchNorm2d(256)
        
        self.encoder_mp3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # ============================ # 
        # Decoder block 1
        self.decoder_mup1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.decoder_conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn1 = nn.BatchNorm2d(128)
        self.decoder_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn2 = nn.BatchNorm2d(128)
        
        # -------------------------# 
        # Decoder block 2
        self.decoder_mup2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.decoder_conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn3 = nn.BatchNorm2d(64)
        self.decoder_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn4 = nn.BatchNorm2d(64)
        
        # -------------------------# 
        # Decoder block 3
        self.decoder_mup3 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.decoder_conv5 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn5 = nn.BatchNorm2d(num_classes)
        self.decoder_conv6 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_bn6 = nn.BatchNorm2d(num_classes)
        
        # -------------------------# 
        # Prediction layer
        
    def forward(self, x):
        # Encoder block 1 pass
        x = F.relu( self.encoder_bn1( self.encoder_conv1(x) ) )
        x = F.relu( self.encoder_bn2( self.encoder_conv2(x) ) )
        x, indices_mp1 = self.encoder_mp1(x)
        
        # Encoder block 2 pass
        x = F.relu( self.encoder_bn3( self.encoder_conv3(x) ) )
        x = F.relu( self.encoder_bn4( self.encoder_conv4(x) ) )
        x, indices_mp2 = self.encoder_mp2(x)
        
        # Encoder block 3 pass
        x = F.relu( self.encoder_bn5( self.encoder_conv5(x) ) )
        x = F.relu( self.encoder_bn6( self.encoder_conv6(x) ) )
        x, indices_mp3 = self.encoder_mp3(x)
        
        # Decoder block 1 pass
        x = self.decoder_mup1(x, indices_mp3)
        x = F.relu( self.decoder_bn1( self.decoder_conv1(x) ) )
        x = F.relu( self.decoder_bn2( self.decoder_conv2(x) ) )
        
        # Decoder block 2 pass
        x = self.decoder_mup2(x, indices_mp2)
        x = F.relu( self.decoder_bn3( self.decoder_conv3(x) ) )
        x = F.relu( self.decoder_bn4( self.decoder_conv4(x) ) )
        
        # Decoder block 3 pass
        x = self.decoder_mup3(x, indices_mp1)
        x = F.relu( self.decoder_bn5( self.decoder_conv5(x) ) )
        x = F.relu( self.decoder_bn6( self.decoder_conv6(x) ) )
        
        return x
# ============================================================================= # 

def load_model(filename): 
    
    model = SegNet(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    
    return model, optimizer
# ============================================================================= # 

def save_model_at_checkpoint(state, epoch):
    filename = "model_at_epoch_" + str(epoch+1) + ".pth.tar"
    torch.save(state, filename)
    
# ============================================================================= # 
# # Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, num_classes):
    
    model.eval()
    class_accuracies = []
    for i in range(num_classes):
        # In each iteration, add an empty list to the main list
        class_accuracies.append([])
    
    predictions = []
    
    with torch.no_grad():
        for index, (data, target) in enumerate(loader):
            print("checking batch " + str(index))
            data = data.to(device=device)
    
            scores = model(data)
            
            for i in range(scores.shape[0]):
                score = scores[i, :, :, :]
                
                prediction = torch.argmax(score.squeeze(), dim=0).cpu().detach().numpy()
                true_label = target[i, :, :].numpy()
                
                predictions.append((prediction, true_label))
                
                for n in range(num_classes):
                # matching = prediction == true_label
                    
                    prediction_class_pos = prediction == n
                    true_label_class_pos = true_label == n
                    matching = prediction_class_pos & true_label_class_pos
                    unmatching = prediction_class_pos ^ true_label_class_pos # xor
                    
                    if true_label_class_pos.sum() != 0:
                        accuracy = (matching.sum() / true_label_class_pos.sum()) - (unmatching.sum() / len(unmatching.flatten()))
                    else:
                        accuracy = 1 - ( (unmatching.sum() / len(unmatching.flatten())) )
                    
                    class_accuracies[n].append(accuracy*100)
          
    model.train()
    return np.array(class_accuracies), predictions

# ============================================================================= # 

def show_overlay(prediction_list, index): 

    plt.figure()
    plt.imshow(prediction_list[index][0])
    plt.figure()
    plt.imshow(prediction_list[index][1])
        
    return

# ============================================================================= # 

torch.cuda.empty_cache()

# Compose transformations 
data_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.ToTensor(),  
    transforms.Normalize( mean = [0.1600, 0.1959, 0.2559], 
                          std=[0.2209, 0.2456, 0.2530] )
    ])   

target_transforms = transforms.Compose([
    transforms.Resize((256,256)),   
    ])   

# Hyperparameters
in_channels = 3
learning_rate = 0.01
batch_size = 16
num_epochs = 35
num_classes = 3
LOAD_MODEL = True

# Load custom dataset
dataset = SegNetDataSet(r'C:\Users\vajra\Documents\GitHub\ML_playground\PyTorch\segnet\archive', 
                        data_transforms=data_transforms, target_transforms=target_transforms)

# Produce test and train sets
train_set, test_set = torch.utils.data.random_split(dataset, [329, 37]) # 90% 10% split between train and test 

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if not LOAD_MODEL: 
    # Initialize network
    model = SegNet(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # Compute the weighting for each class used in the loss
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
    
            # forward
            scores = model(data)
            
            loss = criterion(scores, targets)
            losses[epoch] = loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()
    
            # gradient descent or adam step
            optimizer.step()
            
        # training_accuracies, training_predictions = check_accuracy(train_loader, model, num_classes)
        # testing_accuracies, test_predictions  = check_accuracy(test_loader, model, num_classes)
        
        # avg_training_acc[0][epoch] = training_accuracies[1].mean()
        # avg_training_acc[1][epoch] = training_accuracies[2].mean()
        
        # avg_testing_acc[0][epoch] = testing_accuracies[1].mean()
        # avg_testing_acc[1][epoch] = testing_accuracies[2].mean()
        
    state = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict()}
    save_model_at_checkpoint(state, epoch)
    
else:
    model, optimizer = load_model("model_at_epoch_" + str(num_epochs) + ".pth.tar")
    
# ============================================================================= # 

training_accuracies, training_predictions = check_accuracy(train_loader, model, num_classes)
testing_accuracies, test_predictions  = check_accuracy(test_loader, model, num_classes)

print("\n")
for i in range(1, len(training_accuracies)):
    print("training accuracy for class " + str(i) + " is:")
    print(training_accuracies[i].mean())
    print("test accuracy for class " + str(i) + " is:")
    print(testing_accuracies[i].mean()) 
    