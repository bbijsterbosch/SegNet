from CamVidDataLoader import CamVidLoader
import torch
import torchvision.transforms as transforms
import os
import numpy as np


#Load in the file where the classes for each pixel are specified
#Note: these paths are relative
path_classes      = "CamVidDataSet/CamVid/seg_classes.npy"
#Get the directory where the training images are stored
path_no_lbl_train = "SegNet/CamVidDataSet/CamVid/train"
path_lbl_train   = "SegNet/CamVidDataSet/CamVid/train_labels"
#Get the directory where the validation images are stored
path_no_lbl_val  = "SegNet/CamVidDataSet/CamVid/val"
path_lbl_val     = "SegNet/CamVidDataSet/CamVid/val_labels"

path_no_lbl_test = "SegNet/CamVidDataSet/CamVid/test"
path_lbl_test    = "SegNet/CamVidDataSet/CamVid/test_labels"

classes          = np.load(path_classes)

#Dont forget this line need to transform our dataset into somthing pytorch can use.
transform = transforms.Compose([transforms.ToTensor()])

trainset = CamVidLoader(classes, path_no_lbl_train,  path_lbl_train,  transform=transform)
valset   = CamVidLoader(classes, path_no_lbl_val,    path_lbl_val,    transform=transform)
testset  = CamVidLoader(classes, path_no_lbl_test,   path_lbl_test,   transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True, num_workers=4)
valloader   = torch.utils.data.DataLoader(valset, batch_size=12, shuffle=True, num_workers=4)
testloader  = torch.utils.data.DataLoader(testset, batch_size=12, shuffle=True, num_workers=4)