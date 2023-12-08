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
import wandb

from models import model_dict
from tqdm import tqdm
from b_models import BinaryConnectNet, LiBNet, MoreBiNet


if __name__ == '__main__':
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="LiB-Net",
        
        # track hyperparameters and run metadata
        config={
            "Depth-wise": True,
            "Continue-weight": True,
        }
    )

    # Load CIFAR-10 dataset and apply transformations
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # Initialize the binary neural network based on BinaryConnect
    model_name = 'my_b_net'
    net = LiBNet(num_classes=10, layers=[128, 256, 1024], group_dependency=2).cuda()
    # net = BinaryConnectNet(num_classes=10, layers=[128, 256, 512, 512], activate = 'BiLU', needs_feat=False).cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Saving preparation
    save_dir = './save/teachers_weight'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # Training loop
    best_acc = 0
    for epoch in range(200):  # Change the number of epochs as needed
        running_acc = 0
        running_loss = 0.0
        print(f'Epoch {epoch+1}: Trainning')
        net.train()
        loader_bar = tqdm(trainloader)
        if epoch == 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/5
        elif epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/5

        for inputs, labels in loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, layer in net.named_children():
                # return the weights from binary to continuous when updating
                if name.startswith('conv'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
                elif name.startswith('fc'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
            
            optimizer.step()
            
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
            for name, layer in net.named_children():
                if name.startswith('conv'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
                elif name.startswith('fc'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
            _, predicted = torch.max(outputs, 1)
            round_acc = (predicted == labels).sum().item()
            valling_acc = valling_acc + round_acc
            test_loader_bar.set_postfix(acc=100*round_acc/len(labels))
        print(f'Test Acc: {100*valling_acc/len(testset)}')
        
        wandb.log({'train_acc': 100*running_acc/len(trainset), 'test_acc': 100*valling_acc/len(testset)})
        # save the best model
        if valling_acc > best_acc:
            best_acc = valling_acc

            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': net.state_dict(),
            }
            save_file = os.path.join(save_dir, '{}_best.pth'.format(model_name))
            print('saving the best model!')
            torch.save(state, save_file)
        # print(net.conv1.weight)
        print('-----------------------------------------------------------')
    print('Training Finished')