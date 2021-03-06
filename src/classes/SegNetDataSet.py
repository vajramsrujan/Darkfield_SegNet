# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:00:37 2022

@author: vajra
"""

import glob
import os 
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
# ============================================================================= # 
class SegNetDataSet(Dataset):
    
    def __init__(self, directory, data_transforms=None, target_transforms=None):
        super(SegNetDataSet, self).__init__()
        
        # Grab darkfield image filenames
        self.data = glob.glob(os.path.join(directory,'images','*.png'))
        # Grab truth mask file names
        self.targets = glob.glob(os.path.join(directory,'flattened_masks','*.png'))
        
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        
    def __getitem__(self, index):
        
        # Grab specific data and mask image
        data_path = self.data[index]
        target_path = self.targets[index]
        
        # Convert to PIL Image
        data =  Image.open(data_path)
        target = Image.open(target_path)
        
        # Apply transforms
        if self.data_transforms: 
            data = self.data_transforms(data)
        if self.target_transforms:
            target = self.target_transforms(target)
            
        # Convert mask to tensor
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        
        return data, target

    def __len__(self):
        return len(self.data)