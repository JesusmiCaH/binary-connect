from typing import Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import copy

from models import model_dict
from tqdm import tqdm

class BinaryActivation(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    # @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > 1] = 0
        grad_input[input < -1] = 0
        return grad_input
    
class Binary_conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    def forward(self, input):
        self.weight.clip(-1,1)
        self.bias.clip(-1,1)
        # save the continuous weight here and binarize them
        # the continuous weight will be returned back after backward
        self.saved_weight = self.weight.data
        self.saved_bias = self.bias.data
        w_rand = torch.rand(self.weight.shape)*2 - 1   #binarize strategy: treat weight as the prob of being 1.
        b_rand = torch.rand(self.bias.shape)*2 - 1   
        w_rand, b_rand = w_rand.cuda(), b_rand.cuda()
        if self.training:
            self.weight.data = (self.weight.data - w_rand).sign()
            self.bias.data = (self.bias.data - b_rand).sign()
            # self.weight.data = self.weight.data.sign()
            # self.bias.data = self.bias.data.sign()
        return nn.functional.conv2d(input, self.weight, self.bias, padding=self.padding, groups=self.groups)


class Binary_linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
    def forward(self, input):
        self.weight.clip(-1,1)
        self.bias.clip(-1,1)
        self.saved_weight = self.weight.data
        self.saved_bias = self.bias.data
        w_rand = torch.rand(self.weight.shape)*2 - 1   #binarize strategy: treat weight as the prob of being 1.
        b_rand = torch.rand(self.bias.shape)*2 - 1   
        w_rand, b_rand = w_rand.cuda(), b_rand.cuda()
        if self.training:
            self.weight.data = (self.weight.data - w_rand).sign()
            self.bias.data = (self.bias.data - b_rand).sign()
            # self.weight.data = self.weight.data.sign()
            # self.bias.data = self.bias.data.sign()
        return nn.functional.linear(input, self.weight, self.bias)

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation) -> None:
        super().__init__()
        self.depthwise_conv = Binary_conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel)
        self.pointwise_conv = Binary_conv2d(in_channel, out_channel, kernel_size=1)
        self.activate = activation
        self.downsample = Binary_conv2d(in_channel, out_channel,1)
        # self.depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel)
        # self.pointwise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        # self.activate = activation
        # self.downsample = nn.Conv2d(in_channel, out_channel,1)
    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        # if residual.shape[1] != x.shape[1]:
        #     residual = self.downsample(residual)
        x = self.activate(x)
        return x
    
class BinaryConnectNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(BinaryConnectNet, self).__init__()
        self.conv1 = self._make_layers(1,1,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = self._make_layers(1,64,128)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = Binary_linear(128 * 7 * 7, 1024)
        self.fc2 = Binary_linear(1024, num_classes)   
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        print(x)
        x = BinaryActivation.apply(self.fc1(x))
        x = self.fc2(x)  # No binary activation for the last layer
        return x
    
    def _make_layers(self, depth, in_channel, out_channel):
        layers = []
        layers.append(BasicBlock(in_channel, out_channel, BinaryActivation.apply))

        for i in range(1, depth):
            layers.append(BasicBlock(out_channel, out_channel, BinaryActivation.apply))
        return nn.Sequential(*layers)


if __name__ == '__main__':

    # Load CIFAR-10 dataset and apply transformations
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5))
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # Initialize the binary neural network based on BinaryConnect
    model_name = 'my_b_net'
    net = BinaryConnectNet(num_classes=100).cuda()

    # model_name = 'b_resnet8'
    # net = model_dict[model_name](num_classes=10).cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    # Saving preparation
    save_dir = './save/teachers_weight'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # Training loop
    best_acc = 0
    for epoch in range(50):  # Change the number of epochs as needed
        running_acc = 0
        running_loss = 0.0
        print(f'Epoch {epoch+1}: Trainning')
        net.train()
        loader_bar = tqdm(trainloader)
        if epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/10

        for inputs, labels in loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()

            # print('BEFORE-CHANGE-------------------------------------------------------------')
            # print(net.fc1.bias)
            for name, layer in net.named_children():
                if name.startswith('conv'):
                    for subname, sublayer in list(layer.children())[0].named_children():
                        if subname.endswith('conv'):
                            # if(sublayer.bias.shape[0]<10):
                                # print(sublayer.bias)
                                # print(sublayer.bias.grad)
                            sublayer.weight.data = sublayer.saved_weight
                            sublayer.bias.data = sublayer.saved_bias
                elif name.startswith('fc'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
            
            optimizer.step()
            # print('AFTER-CHANGE-------------------------------------------------------------')
            # for name, layer in net.named_children():
            #     if name.startswith('conv'):
            #         for subname, sublayer in list(layer.children())[0].named_children():
            #             if subname.endswith('conv'):
            #                 if(sublayer.bias.shape[0]<10):
            #                     print(sublayer.bias)
            # print('END-----------------------------------------------------------------------')
            round_acc = (predicted == labels).sum().item()
            running_acc = running_acc + round_acc
            running_loss += loss.item()
            loader_bar.set_postfix(acc=100*round_acc/len(labels))
        print(f'Loss: {running_loss/len(trainset)}, Acc: {100*running_acc/len(trainset)}')

        print('Testing')
        net.eval()
        valling_acc = 0
        test_loader_bar = tqdm(testloader)
        for inputs, labels in test_loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            round_acc = (predicted == labels).sum().item()
            valling_acc = valling_acc + round_acc
            test_loader_bar.set_postfix(acc=100*round_acc/len(labels))
        print(f'Test Acc: {100*valling_acc/len(testset)}')
        
        # save the best model
        if valling_acc > best_acc:
            best_acc = valling_acc

            b_net = copy.deepcopy(net)
            for param in b_net.parameters():
                param.data = param.data.sign()

            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': b_net.state_dict(),
            }
            save_file = os.path.join(save_dir, '{}_best.pth'.format(model_name))
            print('saving the best model!')
            torch.save(state, save_file)
        print('-----------------------------------------------------------')
    print('Training Finished')