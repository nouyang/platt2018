'''
26 Nov 2018
Scratch file for generating minimal "grasp" depth image dataset
Applying Faster-RCNN algorithm
nrw
'''

# TODO: REGRESSION aka How to go from CNN to coordinates+orientation instead of classes?
# TODO: Use Faster-RCNN algorithm

'''
Assumptions
# 300x400 black 'depth image'
# 25x25 square 'blocks' (vary size, orientation) 
# true grasps are parameterized by center and orientation (1 true grasp per square)
# store as numpy array (later pytorch array for GPU)
# ? 2 layer with faster-rcnn using pytorch ?
# start with default pytorch CNN
# start with n = 10 images 

'''

import numpy as np
from PIL import Image, ImageDraw

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas

from torch.utils.data import Dataset, DataLoader

# ----- EXAMPLE -----
# img = Image.new('RGBA', (200,200), 'white')
# idraw = ImageDraw.Draw(img)
# idraw.rectangle((10, 10, 100, 100), fill='blue') #top left corner, and size?
# PIL.ImageDraw.Draw.rectangle(xy, fill=None, outline=None)
#
#    Draws a rectangle.
#    Parameters:
#
#        xy -  Four points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
#        outline - Color to use for the outline.
#        fill -  Color to use for the fill.
#

# https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html
# idraw.polygon([(60,60), (90,60), (90,90), (60,90)], fill="red", outline="green")

# img.save('rectangle.png')
# 
# np.random.rand(m, n)
# np.random.randint(low, high, size=(m,n))
# ----- ------  


# height and width of camera image (in pixels)
H,W = 300,400
# length and width of blocks (fixed for now) 
block_l, block_w = 25, 25 
# block_minl, block_minw = 20, 20

# number of images
num_images = 10  

# list containing PIL Image objects?
img_list = [] 
true_coords = []

# 3 random numbers per image - x,y location, and orientation 
# Y/N: should orientation be chunked by 15 degrees = 24 options? (YES)
# TODO: implement rotated rectangles later 
# TODO: implement blocks that are halfway off-image


for i in range(num_images):
    img = Image.new('RGB', (H, W), 'gray')

    # rand_x = np.random.randint(0, W)
    # rand_y = np.random.randint(0, H)

    rand_x = int(np.random.rand() * (W-block_l))
    rand_y = int(np.random.rand() * (H-block_w))

    # true_coords.append((rand_x, rand_y))
    true_coords.append(np.array((rand_x, rand_y)))


    idraw = ImageDraw.Draw(img)
    # idraw.rectangle((rand_x, rand_y, rand_x+block_l, rand_y+block_w), fill='white')
    idraw.rectangle((rand_x, rand_y, rand_x+block_l, rand_y+block_w), fill='white')

    img_list.append(img)
    # true_coords.append((rand_x, rand_y))
    img.save('./data/rect'+str(i)+'.png')

#print(truth_coords)

#####
# Create pytorch dataloader 
#####


# data loader example 
# 1) create new "rectangle" class which subclasses pytorch "Dataset" class
# 2) use torch.utils.data.DataLoader and specify the batch size etc.
# 3) enumerate to step through the data and update gradients

# 1) create __len__ and __getitem__ functions
class RectDepthImgsDataset(Dataset):
    """Artificially generated depth images dataset"""

    def __init__(self, PIL_images_list, true_coords, transform=None):
        """
        """
        self.images = PIL_images_list 
        self.true_coords = true_coords
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        coords = true_coords[idx]
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            image = self.transform(image) 

        #sample = {'image': image, 'grasp': str(coords[0]) + str(coords[1])}
        # TODO: not numerical classes
        sample = {'image': image, 'grasp': coords}

        return sample


# transformed_dataset =
# FaceLandmarksDataset(csv_file='faces/face_landmarks.csv', root_dir='faces/',
# transform=transforms.Compose([ Rescale(256), RandomCrop(224), ToTensor() ]))

# 2) dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
# 3)  for i, (images, labels) in enumerate(train_loader):


#####
# Set up CNN
#####
# train_dataset = torchvision.datasets.MNIST(root='./data/', train=True,
# transform=transforms.ToTensor(), download=True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 2
batch_size = 5 
learning_rate = 0.001

# Dataset is depth images of rectangular blocks 
train_dataset = RectDepthImgsDataset(PIL_images_list = img_list, true_coords =
                                     true_coords,
                                     transform=transforms.ToTensor())

test_dataset = RectDepthImgsDataset(PIL_images_list = img_list, true_coords =
                                     true_coords, 
                                     transform=transforms.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=False)

'''
# Convolutional neural network (two convolutional layers)
# CNN
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
'''


import torch.nn.functional as F
class Net(nn.Module): #CIFAR is 32x32x3, MNIST is 28x28x1)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(558720, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 558720)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net().to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print('Training model now...')
total_step = len(train_loader)
for epoch in range(num_epochs):

# for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch, sample_batched['image'].size(),
          # sample_batched['landmarks'].size())


    # for i, (images, labels) in enumerate(train_loader):
    for i_batch, sample in enumerate(train_loader):
            # print(i_batch, sample_batched['image'].size(),
          # sample_batched['landmarks'].size())
        print('images')
        print('i of batch: ', i_batch)
        # EXPECT expect size_batch, channels in image, and height+length of image
        print('size of batched images', sample['image'].size())
        # print(type(sample['image']))
        print('grasp')
        # EXPECT 5x1
        print('size of batched grasp labels', sample['grasp'].size())
        # print(type(sample['grasp']))
        print('one batch of grasp labels: ', sample['grasp'])
        images = sample['image']
        labels = sample['grasp']

        images = images.to(device)
        labels = labels.to(device)
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

## ValueError: Expected input batch_size (1) to match target batch_size (5).

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
