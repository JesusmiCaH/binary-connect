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
from b_models import BinaryConnectNet,StudentConnectNet

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
    # Student
    s_net = StudentConnectNet(num_classes=10, layers=[64, 256], activate='BiLU',needs_feat=True).cuda()
    s_net.fc1.weight.data = t_net.fc1.weight.data
    s_net.fc1.bias.data = t_net.fc1.bias.data

    # Define loss function and optimizer
    cls_criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.MSELoss()

    for name in list(s_net.named_parameters())[-2:]:
        print(name[0])
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    conv_optimizer = optim.Adam(list(s_net.parameters())[:-2], lr=1e-2)
    fc_optimizer = optim.Adam(list(s_net.parameters())[-2:], lr=1e-3)

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
            for param_group in conv_optimizer.param_groups:
                param_group['lr'] = param_group['lr']/5
        elif epoch == 160:
            for param_group in conv_optimizer.param_groups:
                param_group['lr'] = param_group['lr']/5

        running_acc = 0
        running_loss = 0.0
        print(f'Epoch {epoch+1}: Trainning')
        s_net.train()
        loader_bar = tqdm(trainloader)

        for inputs, labels in loader_bar:
            if epoch != 0:
                break
            inputs, labels = inputs.cuda(), labels.cuda()
            conv_optimizer.zero_grad()
            feat_s, logit_s = s_net(inputs)
            feat_t, logit_t = t_net(inputs)
            copyed_fc_layer = list(t_net.children())[-1]

            outputs = copyed_fc_layer(feat_s[1].view(feat_s[1].shape[0],-1))

            loss_kd = kd_criterion(feat_s[0], feat_t[0])
            loss_kd.backward()

            _, predicted = torch.max(outputs, 1)

            for name, layer in s_net.named_children():
                if name.startswith('conv'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
            
            conv_optimizer.step()
            
            # round_acc = (predicted == labels).sum().item()

            si = feat_s[1].view(feat_s[1].shape[0],-1).sign()
            ti = feat_t[1].view(feat_s[1].shape[0],-1).sign()
            round_acc = (si == ti).sum().item()
            loader_bar.set_postfix(acc=100*round_acc/si.numel())
            
            running_acc = running_acc + round_acc
            running_loss += loss_kd.item()
            # loader_bar.set_postfix(acc=100*round_acc/len(labels))
        print(f'Loss: {running_loss/len(trainset)}, Acc: {100*running_acc/len(trainset)}')

        print(f'Trainning FC layers')
        running_acc = 0
        running_loss = 0.0
        loader_bar = tqdm(trainloader)

        for inputs, labels in loader_bar:

            inputs, labels = inputs.cuda(), labels.cuda()
            fc_optimizer.zero_grad()
            feat_s, logit_s = s_net(inputs)
            feat_t, logit_t = t_net(inputs)
 
            loss_cls = cls_criterion(logit_s, labels)
            loss_cls.backward()

            outputs = logit_s
            _, predicted = torch.max(outputs, 1)

            for name, layer in s_net.named_children():
                if name.startswith('fc'):
                    layer.weight.data = layer.saved_weight
                    layer.bias.data = layer.saved_bias
            
            fc_optimizer.step()
            
            round_acc = (predicted == labels).sum().item()
            
            running_acc = running_acc + round_acc
            running_loss += loss_cls.item()
            loader_bar.set_postfix(acc=100*round_acc/len(labels))
        print(f'Loss: {running_loss/len(trainset)}, Acc: {100*running_acc/len(trainset)}')

        print('Testing')
        s_net.eval()
        valling_acc = 0
        test_loader_bar = tqdm(testloader)
        for inputs, labels in test_loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            s_net.fc1.weight = t_net.fc1.weight
            s_net.fc1.bias = t_net.fc1.bias
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