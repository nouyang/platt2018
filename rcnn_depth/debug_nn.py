import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print('----- x size: ', x.size(), '------ as input')
        x = self.conv1(x)
        print('----- x size: ', x.size(), '------ after Conv2D 1,6,5')
        x = F.relu(x)
        print('----- x size: ', x.size(), '------ after ReLu')
        x = self.pool(x)
        print('----- x size: ', x.size(), '------ after MaxPool2d(2,2)')
        # x = F.relu(self.conv1(x))

        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
        print('----- x size: ', x.size(), '------')
        # x = F.relu(self.conv2(x))

        x = x.view(-1, self.num_flat_features(x))
        print('----- x size: ', x.size(), '------')
        x = F.relu(self.fc1(x))
        print('----- x size: ', x.size(), '------')
        x = F.relu(self.fc2(x))
        print('----- x size: ', x.size(), '------')
        x = self.fc3(x)
        print('----- x size: ', x.size(), '------')
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
