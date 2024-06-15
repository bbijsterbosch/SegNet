import torch
import torch.utils
import torch.utils.data
from torchsummary import summary
import torchvision.transforms as transforms
import segnet
from DataLoader.CamVidDataLoader import CamVidLoader
import numpy as np
import os
import matplotlib.pyplot as plt


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    #Load in the file where the classes for each pixel are specified
    #Note: these paths are relative
    train = True


    input_channels = 3
    initial_output_channels = 64
    kernel_size = 3
    pool_kernel_size = 2
    num_classes = 32

    model = segnet.SegNet(input_channels, initial_output_channels, num_classes,kernel_size, pool_kernel_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), (3,32,32))
    if train:

        model.init_vgg16_weigths()
        path_classes      = "CamVidDataSet/CamVid/seg_classes.npy"

        #Get the directory where the training and test images are stored
        path_no_lbl_train = "CamVidDataSet/CamVid/train"
        path_lbl_train   = "CamVidDataSet/CamVid/train_labels"
        path_no_lbl_test = "CamVidDataSet/CamVid/val"
        path_lbl_test = "CamVidDataSet/CamVid/val_labels"
        classes = np.load(path_classes)
        #Dont forget this line need to transform our dataset into somthing pytorch can use.
        transform = transforms.Compose([transforms.ToTensor()])

        trainset = CamVidLoader(classes, path_no_lbl_train,  path_lbl_train,  transform=transform)
        valset = CamVidLoader(classes, path_no_lbl_test, path_lbl_test, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=7, shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=7, shuffle = True, num_workers=4)
        trainer_tester = segnet.Train_Test(model, device, trainloader, valloader, classes,  lr=0.1, momentum=0.9, epochs=800) 
        
        trainer_tester.train()
       
    else:
                
        path_classes = "CamVidDataSet/CamVid/seg_classes.npy"
        path_no_lbl_test = "CamVidDataSet/CamVid/test"
        path_lbl_test   = "CamVidDataSet/CamVid/test_labels"
        transform = transforms.Compose([transforms.ToTensor()])
        classes = np.load(path_classes)
        test_set = CamVidLoader(classes, path_no_lbl_test, path_lbl_test, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
        checkpoint = torch.load('model_best.pth')
        model.load_state_dict(checkpoint)
        model.to(device)
        trainer_tester = segnet.Train_Test(model, device, test_loader, test_loader, classes, lr=0.1, momentum=0.9, epochs=800)
        trainer_tester.test()
if __name__ == '__main__':
    main()