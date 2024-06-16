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
        self.vgg16 = models.vgg16(weights='DEFAULT')
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

        self.kaiming_initialization()
        self.init_vgg16_weigths()
        

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
                    x = self.decoder_layers[layer_idx](x)
                    x = F.relu(x)
                    layer_idx += 1
                    x = self.decoder_layers[layer_idx](x)
                    x = F.relu(x)
                    layer_idx += 1
            elif i == 4:
                x = self.decoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
                x = self.decoder_layers[layer_idx](x)
                x = F.relu(x)
                layer_idx += 1
                x = self.decoder_layers[layer_idx](x)              
            else:  # First 3 blocks with 3 convolutions each
                for _ in range(3):
                    x = self.decoder_layers[layer_idx](x)
                    x = F.relu(x)
                    layer_idx += 1
                    x = self.decoder_layers[layer_idx](x)
                    x = F.relu(x)
                    layer_idx += 1

        x_softmax = F.softmax(x, dim=1)

        return x_softmax
    def kaiming_initialization(self):

        for i in range(len(self.decoder_layers)):
            if isinstance(self.decoder_layers[i], nn.ConvTranspose2d):
                w = self.decoder_layers[i].weight
                torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                if self.decoder_layers[i].bias is not None:
                    b = self.decoder_layers[i].bias
                    torch.nn.init.constant_(b, 0)

    def init_vgg16_weigths(self):
        
        layer_idx = 0

        for i in range(len(self.encoder_layers)):
            if isinstance(self.encoder_layers[i], nn.Conv2d):
                # Find the corresponding VGG layer
                while not isinstance(self.vgg16.features[layer_idx], nn.Conv2d):
                    layer_idx += 1

                vgg_layer = self.vgg16.features[layer_idx]

                # Copy weights and biases
                assert self.encoder_layers[i].weight.size() == vgg_layer.weight.size()
                self.encoder_layers[i].weight.data = vgg_layer.weight.data.clone()

                if vgg_layer.bias is not None:
                    assert self.encoder_layers[i].bias.size() == vgg_layer.bias.size()
                    self.encoder_layers[i].bias.data = vgg_layer.bias.data.clone()

                # Freeze parameters
                self.encoder_layers[i].weight.requires_grad = False
                if self.encoder_layers[i].bias is not None:
                    self.encoder_layers[i].bias.requires_grad = False

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
        class_weights = self.compute_class_weights(trainloader, len(classes))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        
    def compute_miou(self, pred,  target, num_classes):
        #Compute the Intersection over Union (IoU) for each class.
        ious = []

        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).sum().item()
            union = pred_inds.sum().item() + target_inds.sum().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / max(union, 1))
        
        ious = np.array(ious)
        mean_iou = np.nanmean(ious)

        return mean_iou
    
    def compute_class_weights(self, dataloader, num_classes):
        counts = np.zeros(num_classes)
        for _, labels in dataloader:
            labels = labels.view(-1)
            for cls in range(num_classes):
                counts[cls] += (labels == cls).sum().item()
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def train(self):
        #Set the optimizer and loss function
        
        is_better = True
        prev_loss = float('inf')
        run_epoch = self.epochs
        writer = SummaryWriter(log_dir='runs/train')
        
        for epoch in range(1, run_epoch + 1):
            self.model.train()
            sum_loss = 0.0
            class_correct = np.zeros(len(self.classes))
            class_total = np.zeros(len(self.classes))
            total_pixels = 0
            correct_pixels = 0
            
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
                predicted = torch.argmax(output_softmax, 1)
                true_labels = torch.argmax(labels, dim=1)
                predicted = predicted.view(-1)
                true_labels = true_labels.view(-1)
                total_pixels += true_labels.numel()
                correct_pixels += (predicted == true_labels).sum().item()
                
                for c in range(len(self.classes)):
                    class_total[c] += (true_labels == c).sum().item()
                    class_correct[c] += ((predicted == c) & (true_labels == c)).sum().item()
                        
                

            
            acc_train = correct_pixels / total_pixels
            class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct, dtype=float), where=class_total!=0)
            class_avg_acc_train = np.mean(class_acc)
                
            loss_epoch = sum_loss/len(self.trainloader)*j
            
            print('Average loss @ epoch {}: {}'.format(epoch, loss_epoch))

            #Evaluate the test set
            self.model.eval()
            test_loss = 0.0
            correct_pixels = 0
            total_pixels = 0
            
            class_correct = np.zeros(len(self.classes))
            class_total = np.zeros(len(self.classes))

            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device)
                    output_softmax = self.model(images)
                    
                    loss = self.loss_fn(output_softmax, labels)
                    test_loss += loss.item()
                    predicted = torch.argmax(output_softmax, 1)
                    true_labels = torch.argmax(labels, dim=1)
                    predicted = predicted.view(-1)
                    true_labels = true_labels.view(-1)
                    total_pixels += true_labels.numel()
                    correct_pixels += (predicted == true_labels).sum().item()
                    
                    for c in range(len(self.classes)):
                        class_total[c] += (true_labels == c).sum().item()
                        class_correct[c] += ((predicted == c) & (true_labels == c)).sum().item()
                        
                

            test_loss /= len(self.testloader)
            acc = correct_pixels / total_pixels
            class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct, dtype=float), where=class_total!=0)
            class_avg_acc = np.mean(class_acc)
            
            
            writer.add_scalars('Loss', {'train': loss_epoch, 'val': test_loss}, epoch)            
            writer.add_scalars('GA', {'train': acc_train, 'val': acc}, epoch)
            writer.add_scalars('CAA', {'train': class_avg_acc_train, 'val': class_avg_acc} , epoch)
            

            is_better = loss_epoch < prev_loss
            if is_better:
                prev_loss = loss_epoch
                torch.save(self.model.state_dict(), "model_best.pth")


            
            torch.cuda.empty_cache()

    def test(self):
        

        test_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        iou_scores = []
        
        class_correct = np.zeros(len(self.classes))
        class_total = np.zeros(len(self.classes))
        classes = self.classes
        for i, data in enumerate(self.trainloader):
            images = data[0].to(self.device, dtype=torch.float32)
            labels = data[1].to(self.device)
            output_softmax = self.model(images)
            loss = self.loss_fn(output_softmax, labels)
            test_loss += loss.item()
            predicted = torch.argmax(output_softmax, 1)
            true_labels = torch.argmax(labels, dim=1)
            predicted = predicted.view(-1)
            true_labels = true_labels.view(-1)
            total_pixels += true_labels.numel()
            correct_pixels += (predicted == true_labels).sum().item()
            
            for c in range(len(self.classes)):
                class_total[c] += (true_labels == c).sum().item()
                class_correct[c] += ((predicted == c) & (true_labels == c)).sum().item()
                
            pred = predicted.cpu()
            label = true_labels.cpu()
            iou_scores.append(self.compute_miou(pred, label, len(self.classes)))
            
            output = output_softmax[0].cpu().detach().numpy()
            output = output.reshape(32, 360, 480)
            output = np.argmax(output, axis=0)
            print(output.shape)
            rgb_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
            for h in range(output.shape[0]):
                for w in range(output.shape[1]):
                    class_idx = output[h, w]
                    rgb_image[h, w] = classes[class_idx]
            plt.imsave(f'outputs/test_{i}.png', rgb_image)
            
            
        test_loss /= len(self.testloader)
        acc = correct_pixels / total_pixels
        class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct, dtype=float), where=class_total!=0)
        class_avg_acc = np.mean(class_acc)
        avg_miou = np.mean(iou_scores)
        
        print(f'Average Loss: {test_loss}')
        print(f'Global Accuracy:{acc}')
        print(f'Class Average Accuracy: {class_avg_acc}')
        print(f'mIoU: {avg_miou}')
  