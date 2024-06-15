import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from itertools import product
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
class SegNet(nn.Module):
    def __init__(self, input_channels, initial_output_channels, num_classes, kernel_size, pool_kernel_size):
        super(SegNet, self).__init__()
        
        self.encoder_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.unpooling_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.vgg16 = models.vgg16(pretrained=True)
        output_channels = initial_output_channels
        
        # Encoder: First 2 blocks with 2 convolutions each
        for _ in range(2):
            self.encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, padding=1))
            self.encoder_layers.append(nn.BatchNorm2d(output_channels))
            self.encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
            self.encoder_layers.append(nn.BatchNorm2d(output_channels))
            self.pooling_layers.append(nn.MaxPool2d(pool_kernel_size, stride=2, return_indices=True))
            input_channels = output_channels
            output_channels *= 2 
            # Double the number of channels for the next block

        # Encoder: Next 2 blocks with 3 convolutions each
        for _ in range(2):
            self.encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, padding=1))
            self.encoder_layers.append(nn.BatchNorm2d(output_channels))
            self.encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
            self.encoder_layers.append(nn.BatchNorm2d(output_channels))
            self.encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
            self.encoder_layers.append(nn.BatchNorm2d(output_channels))
            self.pooling_layers.append(nn.MaxPool2d(pool_kernel_size, stride=2, return_indices=True))
            input_channels = output_channels
            output_channels *= 2


        # Final encoder layer has same input output layers as previous block
        input_channels = output_channels // 2 
        output_channels = input_channels
        self.encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, padding=1))
        self.encoder_layers.append(nn.BatchNorm2d(output_channels))
        self.encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
        self.encoder_layers.append(nn.BatchNorm2d(output_channels))
        self.encoder_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
        self.encoder_layers.append(nn.BatchNorm2d(output_channels))
        self.pooling_layers.append(nn.MaxPool2d(pool_kernel_size, stride=2, return_indices=True))
        

        # Decoder: Last 3 blocks with 3 convolutions each
        self.unpooling_layers.append(nn.MaxUnpool2d(pool_kernel_size, stride=2))
        self.decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(output_channels))
        self.decoder_layers.append(nn.ConvTranspose2d(output_channels, output_channels, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(output_channels))
        self.decoder_layers.append(nn.ConvTranspose2d(output_channels, output_channels, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(output_channels))
        

        for _ in range(2):
            output_channels = input_channels
            self.unpooling_layers.append(nn.MaxUnpool2d(pool_kernel_size, stride=2))
            self.decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, padding=1))
            self.decoder_layers.append(nn.BatchNorm2d(output_channels))
            self.decoder_layers.append(nn.ConvTranspose2d(output_channels, input_channels, kernel_size, padding=1))
            self.decoder_layers.append(nn.BatchNorm2d(input_channels))
            self.decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels//2, kernel_size, padding=1))
            self.decoder_layers.append(nn.BatchNorm2d(output_channels//2))
            input_channels = output_channels // 2

        # Decoder: First 2 blocks with 2 convolutions each
        input_channels = output_channels //2 
        output_channels = input_channels
        self.unpooling_layers.append(nn.MaxUnpool2d(pool_kernel_size, stride=2))
        self.decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(output_channels))
        self.decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels//2, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(output_channels//2))

        output_channels = input_channels // 2
        input_channels = output_channels
        self.unpooling_layers.append(nn.MaxUnpool2d(pool_kernel_size, stride=2))
        self.decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(output_channels))
        self.decoder_layers.append(nn.ConvTranspose2d(output_channels, num_classes, kernel_size, padding=1))
        self.decoder_layers.append(nn.BatchNorm2d(num_classes))

    def forward(self, x):
        pool_indices = []
        sizes = []
        layer_idx = 0

        # Encoder forward pass
        for i in range(2):  # First 2 blocks with 2 convolutions each
            for _ in range(2):
                x = self.encoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
                x = self.encoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
               
            sizes.append(x.size())
            x, indices = self.pooling_layers[i](x)
            pool_indices.append(indices)
        
        
        for i in range(2, 5):
            for _ in range(3):     # Next 3 blocks with 3 convolutions each
                x = self.encoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
                x = self.encoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
            sizes.append(x.size())
            x, indices = self.pooling_layers[i](x)
            pool_indices.append(indices)

        # Decoder forward pass
        layer_idx = 0
        
        for i in range(len(self.unpooling_layers)):
            x = self.unpooling_layers[i](x, pool_indices[-(i+1)], output_size=sizes[-(i+1)])
            if i == 3:  # Last 2 blocks with 2 convolutions each
                for _ in range(2):
                    x = checkpoint(self.decoder_layers[layer_idx], x)
                    x = F.relu(x)
                    layer_idx += 1
                    x = self.decoder_layers[layer_idx](x)
                    x = F.relu(x)
                    layer_idx += 1
            elif i == 4:
                x = checkpoint(self.decoder_layers[layer_idx],x)
                x = F.relu(x)
                layer_idx += 1
                x = self.decoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
                x = self.decoder_layers[layer_idx](x)              
            else:  # First 3 blocks with 3 convolutions each
                for _ in range(3):
                    x = checkpoint(self.decoder_layers[layer_idx],x)
                    x = F.relu(x)
                    layer_idx += 1
                    x = self.decoder_layers[layer_idx](x)
                    x = F.relu(x)
                    layer_idx += 1

        x_softmax = F.softmax(x, dim=1)

        return x_softmax
    def init_vgg16_weigths(self):
        
        layer_idx = 0

        for i in range(len(self.encoder_layers)):
            if (isinstance(self.encoder_layers[i], nn.Conv2d) and isinstance(self.vgg16.features[layer_idx], nn.Conv2d)):
                vgg_layer = self.vgg16.features[layer_idx]
                assert self.encoder_layers[i].weight.size() == vgg_layer.weight.size()
                self.encoder_layers[i].weight.data = vgg_layer.weight.data

                assert self.encoder_layers[i].bias.size() == vgg_layer.bias.size()
                self.encoder_layers[i].bias.data = vgg_layer.bias.data
                layer_idx += 2
                for param in self.encoder_layers[i].parameters():
                    param.requires_grad = False
            
            else:
                layer_idx += 1
            

class Train_Test:
    def __init__(self, model, device, trainloader, testloader, classes, lr, momentum, epochs):
        
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.model = model
        self.classes = classes
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        
    def compute_miou(pred, target, num_classes):
        #Compute the Intersection over Union (IoU) for each class.
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)

        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).sum().item()
            union = pred_inds.sum().item() + target_inds.sum().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / max(union, 1))
        
        ious = np.arrary(ious)
        mean_iou = np.nanmean(ious)

        return mean_iou
    


    def train(self):
        #Set the optimizer and loss function
        self.model.to(self.device)
        self.model.train()
        is_better = True
        prev_loss = float('inf')
        run_epoch = self.epochs
        writer = SummaryWriter(log_dir='runs/experiment1')
        
        for epoch in range(1, run_epoch + 1):
            sum_loss = 0.0

            for j, data in tqdm(enumerate(self.trainloader, 1), total=len(self.trainloader)):
                images, labels = data
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                output_softmax = self.model(images)
                loss = self.loss_fn(output_softmax, labels)
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()
        
                
            loss_epoch = sum_loss/j*self.trainloader.batch_size
            writer.add_scalar('Loss/Train', loss_epoch)
            print('Average loss @ epoch {}: {}'.format(epoch, loss_epoch))

            #Evaluate the test set
            self.model.eval()
            test_loss = 0.0
            correct_pixels = 0
            total_pixels = 0
            iou_scores = []
            
            class_correct = np.zeros(len(self.classes))
            class_total = np.zeros(len(self.classes))

            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device)
                    output_softmax = self.model(images)
                    print(output_softmax.size())
                    loss = self.loss_fn(output_softmax, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(output_softmax, 1)
                    
                    predicted = predicted.view(-1)
                    labels = labels.view(-1)
                    total_pixels += labels.size(0)
                    correct_pixels += (predicted == labels).sum().item()
                    
                    for cls in range(self.classes):
                        class_total[cls] += (labels == cls).sum().item()
                        class_correct[cls] += ((predicted == cls) & (labels == cls)).sum().item()
                        
                    pred = predicted.cpu().numpy()
                    label = labels.cpu().numpy()
                    iou_scores.append(self.compute_miou(pred, label, self.num_classes))

            test_loss /= len(self.testloader)
            acc = correct / total
            class_acc = class_correct / class_total
            class_avg_acc = np.mean(class_acc)
            avg_miou = np.mean(iou_scores)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Global Accuracy', acc, epoch)
            writer.add_scalar('Class Avg. Accuracy', class_avg_acc, epoch)
            writer.add_scalar('mIoU', avg_miou, epoch)

            is_better = loss_epoch < prev_loss
            if is_better:
                prev_loss = loss_epoch
                torch.save(self.model.state_dict(), "model_best.pth")


        self.model.train()
        torch.cuda.empty_cache()

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        classes = np.load('/home/bas/Robotics/CV/SegNet/CamVidDataSet/CamVid/seg_classes.npy')
        for i, data in enumerate(self.trainloader):
            images = data[0].to(self.device, dtype=torch.float32)
            output = self.model(images)[0,:,:]
            output = output.cpu()
            output = output.detach().numpy()
            output = output.squeeze()
            print(output.shape)
            num_classes, height, width = output.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    # Find the class index (the position where the value is 1)
                    class_idx = np.argmax(output[:,i, j])
                    # Get the RGB value for the class index
                    rgb_image[i, j] = classes[class_idx]
            plt.imshow(rgb_image)
            plt.show()
  
   
    
# Example usage
# input_channels = 3
# initial_output_channels = 64
# kernel_size = 3
# pool_kernel_size = 2
# num_classes = 32
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = SegNet(input_channels, initial_output_channels,num_classes, kernel_size, pool_kernel_size)
# # model.init_vgg16_weigths()
# # for layer in model.decoder_layers:
# #     print(layer)
# model.to(device)
# # # layers = 0
# # # for layer in model.encoder_layers:
# # #     print(layer)
# # #     layers += 1

# x = torch.randn(1, 3, 32, 32).to(device)
# # # output = model(x)
# summary(model, (3,32,32))
# # print(output.size())
