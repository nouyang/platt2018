import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# set gradient buffers to zero, set backprops to random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))


# Time to compute the loss
output = net(input)
target = torch.randn(10)  # A random training "target"
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print('loss', loss)


# Inspect backward computations
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# Backprop
net.zero_grad()
print('conv1.bias.grad before backward pass')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


'''
# Update weigths using Stochastic Gradient Descent
# weight = weight - learning_rate * gradient
learning_rate = 0.01
#learnable parameters of a model
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate) # _ postfix means update in place
'''


# create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # perform update
