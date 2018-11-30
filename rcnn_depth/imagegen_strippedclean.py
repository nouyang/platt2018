
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform

from torch.utils.data import Dataset, DataLoader

HEIGHT,WIDTH = 300,400
# length and width of blocks (fixed for now)
block_l, block_w = 25, 25

num_images = 20

img_list = []
true_coords = []

for i in range(num_images):
    img = Image.new('RGB', (HEIGHT, WIDTH), 'gray')

    rand_x = int(np.random.rand() * (WIDTH-block_l))
    rand_y = int(np.random.rand() * (HEIGHT-block_w))

    true_coords.append(np.array((rand_x, rand_y)))

    idraw = ImageDraw.Draw(img)
    idraw.rectangle((rand_x, rand_y, rand_x+block_l, rand_y+block_w), fill='white')

    img_list.append(img)
    img.save('./data/rect'+str(i)+'.png')


class RectDepthImgsDataset(Dataset):
    """Artificially generated depth images dataset"""

    def __init__(self, PIL_images_list, coords, transform=None):
        """
        """
        self.images = PIL_images_list
        self.true_coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = self.images[idx]
        image = io.imread('./data/rect'+str(idx)+'.png')
        print('image dims', image.size)
        image = torch.FloatTensor(image).permute(2,0,1)
        print('image dims', image.size)
        coords = torch.FloatTensor(true_coords[idx])

        if self.transform:
            image = self.transform(image)

        #sample = {'image': image, 'grasp': str(coords[0]) + str(coords[1])}
        sample = {'image': image, 'grasp': coords}
        sample = image, coords

        return sample



# Hyper parameters
num_epochs = 5
num_classes = 2
batch_size = 5
learning_rate = 0.001

# Dataset is depth images of rectangular blocks
train_dataset = RectDepthImgsDataset(PIL_images_list = img_list, coords =
                                     true_coords,
                                     )

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)

const =  16*97*72

import torch.nn.functional as F
class Net(nn.Module): #CIFAR is 32x32x3, MNIST is 28x28x1)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear( const, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        print(x.size())
        x = x.view(-1, 3, 400,300)
        x = self.pool(F.relu(self.conv1(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = x.view(-1, const)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        print(x.size())
        return x


model = Net()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('Training model now...')
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i_batch, (images, labels) in enumerate(train_loader):
        print('images')
        print('i of batch: ', i_batch)
        print('grasp')
        # images = sample[0]
        # labels = sample[1]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        print("----------- LOSS DEBUGGING ----\n")
        print('sizes of inputs:', images.size())
        print('sizes of outputs:', outputs.size())
        print('sizes of labels: ', labels.size(), '\n')
        print('outputs ',outputs, '\n')
        print('labels', labels, '\n')
        print("----------- LOSS DEBUGGING ----\n")
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


