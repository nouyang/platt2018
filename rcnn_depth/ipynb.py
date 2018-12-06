#!/usr/bin/env python
# coding: utf-8

# In[200]:


from PIL import Image, ImageDraw, ImageFont
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

import torch
import torch.nn as nn
from skimage import io
import math

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# In[335]:



IMG_X, IMG_Y = 200,200
# length and width of blocks (fixed for now)
block_l, block_w = 25, 25
num_images = 50

# img_list = []
true_coords = []


# Calc rectangle vertices. makeRectangle() credit Sparkler, stackoverflow, feb 17
def makeRectangle(l, w, theta, offset=(0, 0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x, y) in rectCoords]


# ---- Make depth images ---
for i in range(num_images):
    #orient = 0 # degrees
    img = Image.new('RGB', (IMG_X, IMG_Y), 'black')

    # block_l and _w offset so blocks don't run off edge of image
    rand_x = int(np.random.rand() * (IMG_X-block_l))
    rand_y = int(np.random.rand() * (IMG_Y-block_w))
    orient = int(np.random.rand() * 360)

    true_coords.append(np.array((rand_x, rand_y, orient)))

    rect_vertices = makeRectangle(block_l, block_w, orient, offset=(rand_x,
                                                                    rand_y))

    idraw = ImageDraw.Draw(img)
    idraw.polygon(rect_vertices, fill='white')

    # use a truetype font
    #font = ImageFont.truetype("DejavuSans.ttf", 15)
    #font = ImageFont.truetype("Arial.ttf",14)
    #idraw.text((10, 25), '('+ str(rand_x) + ', ' + str(rand_y) +')')


    img.save('./data/rect'+str(i)+'.png')



class RectDepthImgsDataset(Dataset):
    """Artificially generated depth images dataset"""

    def __init__(self, img_dir, coords, transform=None):
        """
        """
        self.img_dir = img_dir
        self.true_coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.true_coords)

    def __getitem__(self, idx):
        # image = self.images[idx]
        image = io.imread(self.img_dir + '/rect'+str(idx)+'.png')
        image = torch.FloatTensor(image).permute(2, 0, 1) #PIL and torch expect difft orders
        coords = torch.FloatTensor(true_coords[idx])

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'grasp': str(coords[0]) + str(coords[1])}
        sample = {'image': image, 'grasp': coords}
        sample = image, coords

        return sample


# In[387]:


# Hyper parameters

num_epochs = 100
num_classes = 3 # predicting x,y,orientation
batch_size = 15
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available? device: ", device)

# Dataset is depth images of rectangular blocks
train_dataset = RectDepthImgsDataset(img_dir='./data', coords=true_coords)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)

const = 16*47*47
#const = 16*97*72


# 300x300
#torch.Size([4, 300, 300, 3])
#torch.Size([4, 3, 300, 300])
#torch.Size([4, 6, 148, 148])
#torch.Size([4, 16, 72, 72])


# 200x200
#const = 16*47*47*4
#torch.Size([4, 200, 200, 3])
#torch.Size([4, 3, 200, 200])
#torch.Size([4, 6, 98, 98])
#torch.Size([4, 16, 47, 47])

# = 200-4 / 2
# where 4 = 5x5 (conv) -1 = area of image after conv without padding = 200->98
# = 98-4 / 2
# where 4 = 5x5 (conv) -1 = area of image after conv without padding = 98->47
# output is 47x47 image, with 16 output channels


# 400x300
#torch.Size([4, 300, 400, 3])
#torch.Size([4, 3, 400, 300])
#torch.Size([4, 6, 198, 148])
#torch.Size([4, 16, 97, 72])


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


model = Net(IMG_X, IMG_Y)
model = model.to(device)

# ONLY FOR DEBUGGING (check if code runs at all)
#images = iter(train_loader)
##outputs = model(images.next()[0])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[416]:


losses_list = []
print('Training model now...')
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i_batch, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        # magnify orientation loss
        outputs[2] *= 1
        labels[2] *= 1
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        #if (i_batch+1) % 1 == 0:
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1,
                                                                      num_epochs,
                                                                      i_batch+1,
                                                                      total_step,
                                                                      loss.item()))
            losses_list.append(loss.item())


# In[417]:


from IPython.display import display # to display images
#font = ImageFont.truetype("Arial.ttf",14) # OSError: cannot open resource
font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf",14)
def imshow_coord(img, a_label):
    img = torchvision.transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    #draw.text((10, 25), '(' + np.array2string(np.around(a_label.numpy()), separator=', ') + ')')
    x,y,orient = a_label
    rect_vertices = makeRectangle(5, 15, orient, offset=(x,y))
    draw.polygon(rect_vertices, fill='red')
    draw = draw_crosspointer(a_label, draw, 'green', 'white', length=8)
    display(img)

def draw_crosspointer(xy, draw, fillcolor_X='green', fillcolor='white', length=2):
    a,b, orient = tuple(xy)
    draw.line((a-length, b+length, a+length, b-length), fill=fillcolor_X)
    draw.line((a-length, b-length, a+length, b+length), fill=fillcolor_X)
    draw.point((a,b))
    return draw


# In[491]:


import torchvision
import matplotlib.pyplot as plt
import copy
# get some random training images
with torch.no_grad():
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    outputs = model(images)

    # show images
    #imshow_coord(torchvision.utils.make_grid(images), (outputs))


    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  # for j in range(4)))

    loss = criterion(outputs.to(device), labels.to(device))

n =0
#print(len(labels))
print('x,y truth', labels[n])
print('x,y guess by net', outputs[n])

images_copy = copy.deepcopy(images)
print('image size: ', images_copy[n].size())
print(loss)
print('net output: ')
imshow_coord(images_copy[n], outputs[n])
print('truth label: ')
imshow_coord(images_copy[n], labels[n])

#imagePIL = torchvision.transforms.ToPILImage()(images[n])
#print('PIL image size', imagePIL.size)
#imagePIL.save('test.png')
#display(imagePIL)

#display(Image.open('./data/rect'+str(n)+'.png'))


# In[520]:


get_ipython().run_line_magic('matplotlib', 'inline')
#im = Image.open("Mew.jpg")
n = 12
n = np.random.rand()*len(images)
n = int(n)

image = io.imread('./data/rect'+str(n)+'.png')
image_tensor = torch.FloatTensor(image).permute(2, 0, 1) #PIL and torch expect difft orders
coords = torch.FloatTensor(true_coords[n])
with torch.no_grad():
    output = model(image_tensor)

print(true_coords[n])
print(output)
x,y,orient = output.cpu().numpy().flatten()
imshow_coord(image, (x,y, orient))
plt.imshow(image)


# In[334]:


#img = Image.new('RGB', (300,400), 'gray')
sns.set_style("ticks")
n = 0
#imshow_coord(images[n], outputs[n], labels[n])

print(images.size())
a = torchvision.utils.make_grid(images)
print(a.max(), a.min(), a.size())
#a = a / 2 + 0.5     # unnormalize

a = a.cpu().numpy()
a = np.transpose(a,(1,2,0))
print(a.shape)
print(a.ndim)

#ran = a.max() - a.min()
#a = (a/ran )

fig,ax = plt.subplots(1)
# Display the image
ax.imshow(a)
#x,y = labels[0].numpy()
#plt.scatter(x,y, color='g', marker='x', linewidth='1')
labels = labels.cpu()

plt.rcParams['figure.figsize'] = [30,5]
import matplotlib.patches as patches
for i in range(len(labels)):
    x,y, orient = labels[i].numpy()

# class matplotlib.patches.Rectangle(xy, width, height, angle=0.0, **kwargs)[source]

    rect = patches.Rectangle((x + i*IMG_X, y), 1,20,angle=orient, fill=True, color='orange')
    ax.add_patch(rect)
    ax.scatter(x + i*IMG_X, y, color='r', marker='x', linewidth='1')
#plt.imshow(np.transpose(a, (2,0,1)))


# In[290]:


import seaborn as sns
sns.set()
f = plt.subplot()
f.plot(range(len(losses_list)), losses_list)

plt.rcParams['figure.figsize'] = [30, 5]
xposition = np.array(range(num_epoch))  * (total_step)
#print(xposition)
for xc in xposition:
    #plt.axvline(x=xc, color='k', linestyle='--')
    pass



# In[291]:


g = plt.subplot()
trunc = 100
g.plot(range(trunc), losses_list[-trunc:])

plt.show()


# In[210]:


num_epochs = [10,20,50,70,100,150]
num_classes = 3 # predicting x,y,orientation
batch_size = 5
learning_rates = [0.001, 0.005, 0.01, 0.1]

meta_losses_list = []
for i in range(len(num_epochs)):
    num_epoch = num_epochs[i]
    learning_rate = learning_rates[0]
    losses_list = []
    print('Training model now... with: %d num_epoch and %d learning rate' % (num_epoch, learning_rate))

    total_step = len(train_loader)
    for epoch in range(num_epoch):
        for i_batch, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            # magnify orientation loss
            outputs[2] *= 10
            labels[2] *= 10
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (i_batch+1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1,
                                                                          num_epochs,
                                                                          i_batch+1,
                                                                          total_step,
                                                                          loss.item()))
                losses_list.append(loss.item())
    meta_losses_list.append(losses_list)


# In[ ]:


fig = plt.subplots()
trunc = 100
for l in meta_losses_list:
    plt.plot(range(len(l[-trunc:])), l[-trunc:])
plt.legend()
plt.show()

