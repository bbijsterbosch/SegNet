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
    train = False


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
        #Get the directory where the training images are stored
        path_no_lbl_train = "CamVidDataSet/CamVid/train"
        path_lbl_train   = "CamVidDataSet/CamVid/train_labels"
        classes = np.load(path_classes)

        #Dont forget this line need to transform our dataset into somthing pytorch can use.
        transform = transforms.Compose([transforms.ToTensor()])

        trainset = CamVidLoader(classes, path_no_lbl_train,  path_lbl_train,  transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=7, shuffle=True, num_workers=4)
        trainer_tester = segnet.Train_Test(model, device, trainloader, lr=0.1, momentum=0.9, epochs=500)
        
        trainer_tester.train()
        # train_features, train_labels = next(iter(trainloader))
        # print(f"Feature batch shape: {train_features.size()}")
        # print(f"Labels batch shape: {train_labels.size()}")
        # img = train_features[0].squeeze()
        # img = torch.transpose(img, 0, 2)
        # label = train_labels[0].squeeze()
        # label = torch.transpose(label, 0, 2)
        # print(label.shape)

        # # classes = np.load("/home/jzwanen/computer_vision/SegNet/CamVidDataSet/CamVid/seg_classes.npy")
        # # print(classes)
        # # print(type(classes))
        # # print(np.shape(classes))


        # # Ensure the RGB values have the correct shape (32, 3)
        # assert classes.shape == (32, 3), "RGB values should have shape (32, 3)"

        # # Get the shape of the one-hot encoded image
        # height, width, num_classes = label.shape

        # # Check if the number of classes in the one-hot encoded image matches the number of RGB values
        # assert num_classes == 32, "Number of classes in one-hot encoded image should match number of RGB values"

        # # Create an empty array for the RGB image
        # rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        # # Iterate over each pixel and map the one-hot encoded value to the RGB value
        # for i in range(height):
        #     for j in range(width):
        #         # Find the class index (the position where the value is 1)
        #         class_idx = np.argmax(label[i, j])
        #         # Get the RGB value for the class index
        #         rgb_image[i, j] = classes[class_idx]


        # plt.imshow(rgb_image)
        # plt.show()

        # model = model.to(device)
        # trainer_tester.train()
        # train_features, train_labels = next(iter(trainloader))
        # print(f"Feature batch shape: {train_features.size()}")
        # print(f"Labels batch shape: {train_labels.size()}")
        # img = train_features[0].squeeze()
        # img = torch.transpose(img, 0, 2)
        # label = train_labels[0].squeeze()
        # label = torch.transpose(label, 0, 2)
        # print(label.shape)






        # # classes = np.load("/home/jzwanen/computer_vision/SegNet/CamVidDataSet/CamVid/seg_classes.npy")
        # # print(classes)
        # # print(type(classes))
        # # print(np.shape(classes))


        # # Ensure the RGB values have the correct shape (32, 3)
        # assert classes.shape == (32, 3), "RGB values should have shape (32, 3)"

        # # Get the shape of the one-hot encoded image
        # height, width, num_classes = label.shape

        # # Check if the number of classes in the one-hot encoded image matches the number of RGB values
        # assert num_classes == 32, "Number of classes in one-hot encoded image should match number of RGB values"

        # # Create an empty array for the RGB image
        # rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        # # Iterate over each pixel and map the one-hot encoded value to the RGB value
        # for i in range(height):
        #     for j in range(width):
        #         # Find the class index (the position where the value is 1)
        #         class_idx = np.argmax(label[i, j])
        #         # Get the RGB value for the class index
        #         rgb_image[i, j] = classes[class_idx]


        # plt.imshow(rgb_image)
        # plt.show()
    else:
                
        path_classes = "CamVidDataSet/CamVid/seg_classes.npy"
        path_no_lbl_test = "CamVidDataSet/CamVid/test"
        path_lbl_test   = "CamVidDataSet/CamVid/test"
        transform = transforms.Compose([transforms.ToTensor()])
        classes = np.load(path_classes)
        test_set = CamVidLoader(classes, path_no_lbl_test, path_lbl_test, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)
        checkpoint = torch.load('model_best.pth')
        model.load_state_dict(checkpoint)
        model.to(device)
        trainer_tester = segnet.Train_Test(model, device, test_loader, lr=0.1, momentum=0.9, epochs=150)
        trainer_tester.test()
if __name__ == '__main__':
    main()