# Box prediction
## Trying again! 
from PIL import Image, ImageDraw
import numpy as np

import torch
import torch.nn as nn
from skimage import io
import math

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os


IMG_X, IMG_Y = 200,200 
# length and width of blocks (fixed for now)
block_l, block_w = 20, 30


# Calc rectangle vertices. makeRectangle() credit Sparkler, stackoverflow, feb 17
def makeRectangle(l, w, theta, offset=(0, 0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x, y) in rectCoords]

# ---- Make depth images ---
def make_dataset(dirname, num_images):
    true_coords = []
    newpath = './' + dirname  
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(newpath)
    for i in range(num_images):
        #orient = 0 # degrees
        img = Image.new('RGB', (IMG_X, IMG_Y), 'black')

        # block_l and _w offset so blocks don't run off edge of image 
        rand_x = int(np.random.rand() * (IMG_X-2*block_l)) + block_l
        rand_y = int(np.random.rand() * (IMG_Y-2*block_w)) + block_w
        orient = int(np.random.rand() * 180)  # .random() is range [0.0, 1.0).
        orient = math.radians(orient) # math.cos takes radians!

        true_coords.append(np.array((rand_x, rand_y, orient)))

        rect_vertices = makeRectangle(block_l, block_w, orient, offset=(rand_x,
                                                                        rand_y))

        idraw = ImageDraw.Draw(img)
        idraw.polygon(rect_vertices, fill='white')

        # use a truetype font
        #font = imagefont.truetype("dejavusans.ttf", 15)
        #font = imagefont.truetype("arial.ttf",14)
        #idraw.text((10, 25), '('+ str(rand_x) + ', ' + str(rand_y) +')')
        img.save(newpath + '/rect'+str(i)+'.png')
    return true_coords


class RectDepthImgsDataset(Dataset):
    """Artificially generated depth images dataset"""

    def __init__(self, img_dir, coords, transform=None):
        self.img_dir = img_dir
        self.true_coords = coords
        self.transform
        
        = transform

    def __len__(self):
        #print('true coord len', len(self.true_coords))
        return len(self.true_coords)

    def __getitem__(self, idx):
        # image = self.images[idx]
        image = io.imread(self.img_dir + '/rect'+str(idx)+'.png')
        image = torch.FloatTensor(image).permute(2, 0, 1) #PIL and torch expect difft orders
        coords = torch.FloatTensor(self.true_coords[idx])

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'grasp': str(coords[0]) + str(coords[1])}
        sample = {'image': image, 'grasp': coords}
        sample = image, coords

        return sample


class Net(nn.Module):  # CIFAR is 32x32x3, MNIST is 28x28x1)
    def __init__(self, IMG_X, IMG_Y):
        super(Net, self).__init__()
        self._imgx = IMG_X
        self._imgy = IMG_Y
        _pool = 2
        _stride = 5
        _outputlayers = 16

        def _calc(val):
            layer_size = (val- (_stride-1)) / _pool
            return layer_size

        #print(self._imgx)
        self._const = _calc(_calc(self._imgx))
        self._const *= _calc(_calc(self._imgy))
        self._const *= _outputlayers
        #print(self._const)
        self._const = int(self._const)

        self.conv1 = nn.Conv2d(3, 6, _stride).to(device)
        self.pool = nn.MaxPool2d(_pool, _pool).to(device)
        self.conv2 = nn.Conv2d(6, _outputlayers, _stride).to(device)
        self.fc1 = nn.Linear(self._const, 120).to(device)
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, num_classes).to(device)

    def forward(self, x):
        #print(x.size())
        x = x.to(device)
        x = x.view(-1, 3, IMG_X, IMG_Y)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._const)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def run_dataset_creation():
    train_truth = make_dataset('data', 500)
    print(len(train_truth))
    test_truth = make_dataset('./data/test', 300)
        # to things

    batch_size = 15 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA available? device: ", device)

    # Dataset is depth images of rectangular blocks
    train_dataset = RectDepthImgsDataset(img_dir='./data', coords=train_truth)
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)

    test_dataset = RectDepthImgsDataset(img_dir='./data/test', coords=test_truth)
    # Data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                              shuffle=True)


def model_dataset_creation():
    num_classes = 3 # predicting x,y,orientation
    learning_rate = 0.001

    # ONLY FOR DEBUGGING (check if code runs at all)
    #images = iter(train_loader)
    ##outputs = model(images.next()[0])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = Net(IMG_X, IMG_Y)
    model = model.to(device)



def train_dataset():
    num_epochs = 50 

    losses_list = []
    ct = 0

    print('Training model now...')
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i_batch, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            print('This is batch', i_batch, ' with len images ', len(images))

            # Forward pass
            outputs = model(images).to(device)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            #if (i_batch+1) % 1 == 0:
            if (epoch+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(\
                    epoch+1, num_epochs, i_batch+1, total_step, loss.item()))
            losses_list.append(loss.item())


def view_results():
    criterion = nn.MSELoss()


    print(len(test_loader))
    print(len(train_loader))
    print(len(test_loader))
    print(len(test_loader))

    with torch.no_grad():
        total_err = 0
        n_total = 0
        for i_batch, (images, labels) in enumerate(test_loader):
        #for i_batch, (images, labels) in enumerate(train_loader):
            print('i_batch', i_batch)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            diff = outputs - labels
            diff = torch.sum(diff, 0) #column sum
            total_err += diff
    print(n_total * batch_size)


