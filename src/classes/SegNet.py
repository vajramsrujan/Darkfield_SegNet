#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:28:23 2022

@author: srujanvajram
"""

# Imports
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import nn  # All neural network modules
# ============================================================================= # 

#  SegNet
class SegNet(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        
        # -------------------------# 
        # Encoder block 1
        
        # Convolution layer 1
        self.encoder_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Batch norm layer 1
        self.encoder_bn1 = nn.BatchNorm2d(64)
        # Convolution layer 2
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Batch norm layer 2
        self.encoder_bn2 = nn.BatchNorm2d(64)
        
        # Max pool
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
        # Max Unpool layer
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
        
        '''
            Defines the forward pass sequence of input to the model
        '''
        
        # Encoder block 1 pass
        # Pass through conv 1, batch norm 1 and relu operator
        x = F.relu( self.encoder_bn1( self.encoder_conv1(x) ) )
        # Pass through conv 2, batch norm 2 and relu operator
        x = F.relu( self.encoder_bn2( self.encoder_conv2(x) ) )
        # Memoize indics of max pool (used later for max Unpool)
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
        # Pass indices of max pool layer 1 to unpool
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
    

    