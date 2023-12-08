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
from b_models import BinaryConnectNet,StudentConnectNet,DistillKL


if __name__ == '__main__':

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
    t_model_name = 'my_b_net'
    t_net = BinaryConnectNet(num_classes=10, layers=[128,256,512,256], activate='BiLU',needs_feat=True).cuda()
    # Loading preparation
    t_save_dir = './save/teachers_weight'
    if not os.path.isdir(t_save_dir):
        os.mkdir(t_save_dir)
    t_net.load_state_dict(torch.load(os.path.join(t_save_dir, '{}_best.pth'.format(t_model_name)))['model'])
    t_net.eval()
    s_net = StudentConnectNet(num_classes=10, layers=[64, 256], activate='BiLU',needs_feat=True).cuda()

    # Define loss function and optimizer
    cls_criterion = nn.CrossEntropyLoss()
    kd_criterion = DistillKL(4)


    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(s_net.parameters(), lr=1e-3)

    # Saving preparation
    model_name = 'my_sb_net'
    s_save_dir = './save/students_weight'
    if not os.path.isdir(s_save_dir):
        os.mkdir(s_save_dir)
    
    # Training loop
    best_acc = 0
    for epoch in range(200):  # Change the number of epochs as needed
        if epoch == 80:
            # 如果成了记得把这里全改一下
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/5
        elif epoch == 160:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/5

        running_acc = 0
        running_loss = 0.0
        print(f'Epoch {epoch+1}: Trainning')
        s_net.train()
        loader_bar = tqdm(trainloader)

        for inputs, labels in loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            feat_s, logit_s = s_net(inputs)

            feat_t, logit_t = t_net(inputs)
            outputs = logit_s

            loss_div = kd_criterion(logit_s, logit_t)
            loss_div.backward()
            # loss = cls_criterion(outputs, labels)
            # loss.backward()
            
            _, predicted = torch.max(outputs, 1)

            for name, layer in s_net.named_children():
                if name.startswith('conv'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
                if name.startswith('fc'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
            
            optimizer.step()
            
            round_acc = ( predicted == labels).sum().item()
            running_acc = running_acc + round_acc
            running_loss += loss_div.item()
            loader_bar.set_postfix(acc=100*round_acc/len(labels))
        print(f'Loss: {running_loss/len(trainset)}, Acc: {100*running_acc/len(trainset)}')

        print('Testing')
        s_net.eval()
        valling_acc = 0
        test_loader_bar = tqdm(testloader)
        for inputs, labels in test_loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            _, outputs = s_net(inputs)
            for name, layer in s_net.named_children():
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
        
        # save the best model
        if valling_acc > best_acc:
            best_acc = valling_acc

            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': s_net.state_dict(),
            }
            save_file = os.path.join(s_save_dir, '{}_best.pth'.format(model_name))
            print('saving the best model!')
            torch.save(state, save_file)
        # print(net.conv1.weight)
        print('-----------------------------------------------------------')
    print('Training Finished')