import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

from models import model_dict
from tqdm import tqdm

# Define the binary network
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
        b_weight = self.weight.sign()
        b_bias = self.bias.sign()
        return nn.functional.conv2d(input, b_weight, b_bias, padding=self.padding, groups=self.groups)

class BinaryConnectNet(nn.Module):
    def __init__(self):
        super(BinaryConnectNet, self).__init__()
        self.conv1 = nn.Sequential(
            Binary_conv2d(3, 3, kernel_size=3, padding=1, groups=3),  # Depthwise convolution
            Binary_conv2d(3, 128, kernel_size=1),  # Pointwise convolution
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Sequential(
            Binary_conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise convolution
            Binary_conv2d(128, 256, kernel_size=1),  # Pointwise convolution
        )
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x, need_feat=False):
        x = self.pool(BinaryActivation.apply(self.conv1(x)))
        x = self.pool(BinaryActivation.apply(self.conv2(x)))
        f1 = x
        x = x.view(x.size(0), -1)
        x = BinaryActivation.apply(self.fc1(x))
        x = self.fc2(x)  # No binary activation for the last layer
        if need_feat:
            return f1, x
        else:
            return x


if __name__ == '__main__':
    # Load CIFAR-10 dataset and apply transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # Initialize the binary neural network based on BinaryConnect
    # net = BinaryConnectNet()
    t_save_dir = './save/teachers_weight'
    t_model_name = 'resnet32x4'
    teacher_net = model_dict[t_model_name](num_classes=10).cuda()
    teacher_net.load_state_dict(torch.load(os.path.join(t_save_dir, '{}_best.pth'.format(t_model_name)))['model'])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=4e-4)

    # Saving preparation
    save_dir = './save/teachers_weight'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # Training loop
    best_acc = 0
    for epoch in range(20):  # Change the number of epochs as needed
        running_acc = 0
        running_loss = 0.0
        loader_bar = tqdm(trainloader)
        for inputs, labels in loader_bar:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_acc = running_acc + (predicted == labels).sum().item()
            running_loss += loss.item()
            loader_bar.set_postfix(loss=loss.item())
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainset)}, Acc: {100*running_acc/len(trainset)}')

        # save the best model
        if running_acc > best_acc:
            best_acc = running_acc
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': net.state_dict(),
            }
            save_file = os.path.join(save_dir, '{}_best.pth'.format(model_name))
            print('saving the best model!')
            torch.save(state, save_file)

    print('Training Finished')