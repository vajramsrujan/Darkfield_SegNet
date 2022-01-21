# Darkfield_SegNet


## Summary
A complete breakdown of the entire project can be found in the **semantic segmentation jupyter notebook** titled *Semantic_SegNet.ipynb.*
This is a semantic segmentation project I built in PyTorch to segment red blood cells from bacteria in darkfield microscopy images. You can also find a pdf version in the same directory should you have trouble viewing notebooks. 

## Description: 
I've been interested in constructing a semantic segmenter for microscopy image data for a while. My segmenter is built directly off the SegNet encoding-decoding architecture as described in the 2015 SegNet paper from the University of Cambridge. The full details of this architecture can be found at: https://arxiv.org/pdf/1511.00561.pdf

## A note about the dataset
For obvious reasons, I can not push the dataset onto Github given its size. Instead, I provide a link to a google drive where the pre-processed dataset is ready to download: 

https://drive.google.com/drive/folders/1RQVSvPGG5Sh2WUbT0TmfgHgkFnZ9UPKW?usp=sharing

IMPORTANT:
***Once these folders are downloaded, drop them into the 'archive' folder found in the src folder. Also, make sure to change the directory string when initializing the dataset in main.py to point to the archive folder (line 49.)***


## Running SegNet
I have pre-saved a trained model laballed as a .tar file. Currently, you can test this model by setting the LOAD_MODEL parameter to True in main.py. I've included some helpful comments for guidance. 
