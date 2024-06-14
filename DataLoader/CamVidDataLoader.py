import torch
from torch.utils.data import Dataset
import os
from skimage import io
from skimage.transform import rescale
import time
import numpy as np
import matplotlib.pyplot as plt


class CamVidLoader(Dataset):

    def __init__(self, classes, no_lbl_dir, lbl_dir, transform=None): 
        # classes: (np ndarray) (K, 3) array of RGB values of K classes
        # raw_dir: (directory) Folder directory of raw input image files
        # lbl_dir: (directory) Folder directory of labeled image files:
        self.classes    = classes
        self.no_lbl_dir = no_lbl_dir
        self.lbl_dir    = lbl_dir
        self.transform  = transform

        #Do not include any hidden files
        self.no_lbl_list_img = [file for file in os.listdir(self.no_lbl_dir) if not file.startswith('.')]
        self.lbl_list_img    = [file for file in os.listdir(self.lbl_dir) if not file.startswith('.')]
        
    def one_hot(self, image):
        
        #Used for pixel-wise conversion of labeled images to its respective classes
        #Output is a one-hot encoded tensor of (M, N, K) dimensions, MxN resolution, K channels (classes)
        #K = 32 for CamVid


        output_shape = (image.shape[0], image.shape[1], self.classes.shape[0])
        output = np.zeros(output_shape)
        
        # Loop through each class to see if it present in the image
        for c in range(self.classes.shape[0]):
            label = np.nanmin(self.classes[c] == image, axis=2)
            # Change the corresponding 0 to a 1 if the class is present in the image 
            output[:, :, c] = label
        
        return output

    def __len__(self):
        return len(self.no_lbl_list_img)
    
    def __getitem__(self, index):
        #Get the path to the image(no label) with the given index
        no_lbl_img_path = os.path.join(self.no_lbl_dir, self.no_lbl_list_img [index])
        # Read in the image
        image_raw = io.imread(no_lbl_img_path)
       
        image_resized = rescale(image_raw, 0.5, anti_aliasing=True, channel_axis=2)
       
        # Get the path the the same image with label
        lbl_img_path    = os.path.join(self.lbl_dir,    self.lbl_list_img[index])
        # Read in the image in the file
        image_label    = io.imread(lbl_img_path)
        image_label_resized = rescale(image_label, 0.5, preserve_range=True, anti_aliasing=False, order=0, channel_axis=2)
        
        # Encode each pixel with its one hot encoding that corresponds to the class
        one_hot_label =  self.one_hot(image_label_resized)
        

        if self.transform:
            image_resized = self.transform(image_resized)
            one_hot_label = self.transform(one_hot_label)
        
        data = (image_resized, one_hot_label)

        return data
    
#print(DataLoaderEx.lbl_list_img)

