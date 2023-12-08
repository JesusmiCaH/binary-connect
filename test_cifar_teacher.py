from typing import Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

from models import model_dict
from tqdm import tqdm
from b_models import BinaryConnectNet

if __name__ == '__main__':
    # Load CIFAR-10 dataset and apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # Initialize the binary neural network based on BinaryConnect
    model_name = 'my_b_net'
    net = BinaryConnectNet(num_classes=10, layers=[128,256,512,256], activate='BiLU', needs_feat=False).cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=4e-4)

    # Loading preparation
    save_dir = './save/teachers_weight'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    net.load_state_dict(torch.load(os.path.join(save_dir, '{}_best.pth'.format(model_name)))['model'])

    for name, layer in net.named_children():
        if name.startswith('conv'):
            layer.weight.data = layer.weight.data.sign()
            layer.bias.data = layer.bias.data.sign()
        elif name.startswith('fc'):
            layer.weight.data = layer.weight.data.sign()
            layer.bias.data = layer.bias.data.sign()
        elif name.startswith('bn'):
            pass
    # for name, para in net.named_parameters():
    #     print(name)
    #     print(para)
    print(torch.load(os.path.join(save_dir, '{}_best.pth'.format(model_name)))['epoch'])
    # Testing loop
    net.eval()
    with torch.no_grad():
        running_acc = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            running_acc += (predicted == labels).sum().item()
            # print((predicted == labels).sum().item())
        print(f'Acc: {100*running_acc/len(testset)}')