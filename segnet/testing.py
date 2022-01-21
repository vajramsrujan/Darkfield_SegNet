#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:00:55 2022

@author: srujanvajram
"""

import numpy as np
from PIL import Image 
from matplotlib import pyplot as plt
import torch

# for i in range(1,367): 
    
#     image = Image.open('archive/images/' + str(i).zfill(3) + '.png').convert('L')
#     image.save('archive/greyscale_images/' + str(i).zfill(3) + '.png')

# image = Image.open('archive/flattened_masks/' + "036" + '.png')
# im2arr = np.array(image)
# arr2im = Image.fromarray(im2arr).convert('RGB')
# arr2im.save("036_coloredmask.png")

# class_1 = class_2 = class_3 = 0
# for i in range(1,367):
#     image = Image.open('archive/flattened_masks/' +  str(i).zfill(3) + '.png')
#     imm2arr = np.array(image)
#     class_1 += len(np.where(imm2arr==0)[0])
#     class_2 += len(np.where(imm2arr==1)[0])
#     class_3 += len(np.where(imm2arr==2)[0]) 
    
    
image = Image.open('archive/matched_images/' + "training_image_1" + '.png')
image = image.resize((256,256), resample=0)

mask = Image.open('archive/matched_images/' + "training_pred_1" + '.png')
mask = mask.resize((256,256), resample=0)


plt.figure()
# plt.imshow(image)
plt.imshow(mask, alpha=0.5)