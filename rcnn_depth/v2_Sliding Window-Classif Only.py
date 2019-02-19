# # Sliding Window
# Feb 2019
#
import matplotlib.patches as patches
import seaborn as sns
import copy
import torchvision
from IPython.display import display  # to display images
from PIL import Image, ImageDraw
import numpy as np

import torch
import torch.nn as nn
from skimage import io
import math

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt

import time
import os


# In[2]:


IMG_X, IMG_Y = 200, 200
# length and width of blocks (fixed for now)
block_l, block_w = 20, 30


# Make dataset -------------------------------------------------------


def makeRectangle(l, w, theta, offset=(0, 0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l / 2.0, w / 2.0), (l / 2.0, -w / 2.0),
                  (-l / 2.0, -w / 2.0), (-l / 2.0, w / 2.0), ]
    return [(c * x - s * y + offset[0],
             s * x + c * y + offset[1]) for (x, y) in rectCoords]


# ---- Make depth images ---
def make_dataset(dirname, num_images):
    true_coords = []
    newpath = "./" + dirname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(newpath)
    for i in range(num_images):
        # orient = 0 # degrees
        img = Image.new("RGB", (IMG_X, IMG_Y), "black")

        # block_l and _w offset so blocks don't run off edge of image
        rand_x = int(np.random.rand() * (IMG_X - 2 * block_l)) + block_l
        rand_y = int(np.random.rand() * (IMG_Y - 2 * block_w)) + block_w
        orient = int(np.random.rand() * 180)  # .random() is range [0.0, 1.0).
        orient = math.radians(orient)  # math.cos takes radians!

        true_coords.append(np.array((rand_x, rand_y, orient)))

        rect_vertices = makeRectangle(
            block_l, block_w, orient, offset=(rand_x, rand_y))

        idraw = ImageDraw.Draw(img)
        idraw.polygon(rect_vertices, fill="white")

        img.save(newpath + "/rect" + str(i) + ".png")
    return true_coords


# NOTE Define size of dataset
# train_truth = make_dataset("data", 1500)
# print(len(train_truth))
# test_truth = make_dataset("./data/test", 300)

#np.save("train_truth.npy", train_truth)
#np.save("test_truth.npy", test_truth)


train_truth = np.load("train_truth.npy")
test_truth = np.load("test_truth.npy")  # loading the training and testing data


# Define Dataloader -------------------------------------------------------
class RectDepthImgsDataset(Dataset):
    """Artificially generated depth images dataset"""

    def __init__(self, img_dir, coords, transform=None):
        """
        """
        self.img_dir = img_dir
        self.true_coords = coords
        self.transform = transform

        STEPSIZE = 50  # todo make this input arguments
        WINDOWSIZE = (100, 100)
        MARGIN_PX = 15
        self.step = STEPSIZE
        self.cropSize = WINDOWSIZE
        self.detectMargin = MARGIN_PX

    def __len__(self):
        # print('true coord len', len(self.true_coords))
        return len(self.true_coords)

    def __getitem__(self, idx):
        # image = self.images[idx]
        image = io.imread(self.img_dir + "/rect" +
                          str(idx) + ".png", as_grey=True)
        # image = torch.FloatTensor(image).permute(2, 0, 1)  # PIL and torch expect difft orders
        coords = torch.FloatTensor(self.true_coords[idx])

        if self.transform:
            image = self.transform(image)

        crops, labels = self.makeCrops(
            image, self.step, self.cropSize, coords, self.detectMargin
        )

        sample = image, torch.FloatTensor(labels), coords
        return sample

    def makeCrops(self, image, stepSize, windowSize, rectCenter, detectMargin):
        """
        Returns image crops, as well as T/F for those crops
        """
        crops = []
        c_x, c_y, theta = rectCenter
        margin = detectMargin
        truths = []
        for y in range(0, image.shape[0] - windowSize[0] + 1, stepSize):
            for x in range(0, image.shape[1] - windowSize[1] + 1, stepSize):
                end_x, end_y = x + windowSize[1], y + windowSize[0]
                hasRect = (x + margin < c_x < end_x - margin) and (
                    y + margin < c_y < end_y - margin)
                crops.append(image[y:end_y, x:end_x])
                truths.append(hasRect)
        # print('length of truths in makeCrops', len(truths))
        return crops, truths


# -- Define NN -------------------------------------------------------

class myNet(nn.Module):  # CIFAR is 32x32x3, MNIST is 28x28x1)
    def __init__(self, IMG_X, IMG_Y):
        """ We need the image width and height to determine CNN layer sizes
        """
        super(myNet, self).__init__()
        self._imgx = IMG_X
        self._imgy = IMG_Y
        _pool = 2
        _stride = 5
        _outputlayers = 16

        STEPSIZE = 50  # todo make this input arguments
        WINDOWSIZE = (100, 100)
        self.step = STEPSIZE
        self.cropSize = WINDOWSIZE
        self.numCrops = 0
        # T/F for now

        # calculate number of crops
        for x in range(0, IMG_Y - WINDOWSIZE[0] + 1, STEPSIZE):
            for y in range(0, IMG_X - WINDOWSIZE[1] + 1, STEPSIZE):
                self.numCrops += 1
        # TODO: but our net wants everything to be the same size... pad with 0s?

        def _calc(val):  # use to calculate layer sizes
            layer_size = (val - (_stride - 1)) / _pool
            return layer_size

        # print(self._imgx)
        # self._const = _calc(_calc(self._imgx))
        # self._const *= _calc(_calc(self._imgy))
        self._const = _calc(_calc(self.cropSize[0]))
        self._const *= _calc(_calc(self.cropSize[1]))
        self._const *= _outputlayers
        self._const = int(self._const)

        # batch, 3 input image channels (RGB), 6 output channels, 5x5 square convolution
        # NOTE: we switched to 1 input channel
        self.conv1 = nn.Conv2d(self.numCrops, 6, _stride).to(device)
        self.pool = nn.MaxPool2d(_pool, _pool).to(device)
        self.conv2 = nn.Conv2d(6, _outputlayers, _stride).to(device)
        self.fc1 = nn.Linear(self._const, 120).to(device)
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, self.numCrops).to(device)
        self.sigmoid = nn.Sigmoid()
        # (* N crops, if we wanted output for all # windows at once)
        # TODO: batch normalization  self.bn = nn.BatchNorm2d()

    #
    # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
    # self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
    # Class prediction convolutions (predict classes in localization boxes)
    # self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.to(device)
        # TODO: presumably by doing this i lose some of the multithread goodness
        batch_images = x

        all_crops = []
        for img in batch_images:
            crops = self.makeCrops(img, self.step, self.cropSize)
            all_crops.append(crops)
            # all_crops.append(crops)
        all_crops = torch.stack(all_crops)
        # gray=1
        # we want, batch_size, [100x100]*numcrops 4 dims
        # TODO is this even right # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! <<<<-----------------------
        x = all_crops.view(-1, self.numCrops, self.cropSize[0], self.cropSize[1]).to(
            device
        )

        # NOTE we are shifting to gray scale now!!!
        x = self.pool(F.relu((self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._const)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def makeCrops(self, image, stepSize, windowSize):
        """
        Returns a generator of cropped boxes(the top left x, y, the image data)
        """
        image = image.type(torch.FloatTensor).to(device)
        crops = []

        # TODO: look into ordering, why it's y,x !
        for y in range(0, image.shape[0] - windowSize[0] + 1, stepSize):
            for x in range(0, image.shape[1] - windowSize[1] + 1, stepSize):
                end_x, end_y = x + windowSize[1], y + windowSize[0]
                # print('This is the x and y used: ', x, '; ', y)
                crops.append(image[y:end_y, x:end_x])
        crops = torch.stack(crops)
        # self.numCrops=len(crops)
        return crops


# -- Utility fxn -------------------------------------------------------
# Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best
):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "loss": loss,
        "best_loss": best_loss,
        "model": model,
        "optimizer": optimizer,
    }
    filename = "checkpoint_v2sliding.pth.tar"
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, "BEST_" + filename)


# -- Define Train and Valid fxn -------------------------------------------------------

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    start = time.time()

    for i_batch, (images, labels, coords) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)

        # Backward prop
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i_batch,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                )
            )
        del outputs, images, labels  # free some memory since their histories may be stored


def validate(val_loader, model, criterion):
    """
    One epoch's validation.
    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i_batch, (images, labels, coords) in enumerate(train_loader):

            # Move to default device
            images = images.to(device)
            labels = labels.to(device)

            # Forward prop.
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print(
                    "[{0}/{1}]\t"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        i_batch, len(val_loader), batch_time=batch_time, loss=losses
                    )
                )

    print("\n * LOSS - {loss.avg:.3f}\n".format(loss=losses))

    return losses.avg


# -- Load data -------------------------------------------------------
batch_size = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available? device: ", device)


# Dataset is depth images of rectangular blocks
train_dataset = RectDepthImgsDataset(img_dir="./data", coords=train_truth)

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)

test_dataset = RectDepthImgsDataset(img_dir="./data/test", coords=test_truth)

# Data loader
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)


# -- Hyperparamaters -------------------------

num_epochs = 30  # number of epochs to run without early-stopping
learning_rate = 0.001

start_epoch = 0  # start at this epoch
# number of epochs since there was an improvement in the validation metric
epochs_since_improvement = 0
best_loss = 1.0  # assume a high loss at first
workers = 4  # number of workers for loading data in the DataLoader

model = myNet(IMG_X, IMG_Y)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
#                             momentum=momentum, weight_decay=weight_decay)

print_freq = 25  # print training or validation status every __ batches


def main():
    """
    Training and validation.
    """

    global epochs_since_improvement, start_epoch, best_loss, epoch, checkpoint

    print("Training model now...")

    # -- Begin training -------------------------

    for epoch in range(num_epochs):
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        # One epoch's validation
        val_loss = validate(val_loader=test_loader,
                            model=model, criterion=criterion)

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer,
                        val_loss, best_loss, is_best)

    print("All ready!")

    # -- Check the results -------------------------------------------------------


# -- Utility ---------------------------------------------
def makeCrops(image, stepSize, windowSize, true_center):
    """
    """
    image = image.type(torch.FloatTensor)
    crops = []
    truths = []
    c_x, c_y, orient = true_center
    # TODO: look into otdering, why it's y,x !
    margin = 15
    # --> is x, but is the column
    # to slide horizontally, y must come first
    for y in range(0, image.shape[0] - windowSize[0] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[1] + 1, stepSize):
            end_x, end_y = x + windowSize[1], y + windowSize[0]
            hasRect = (x + margin < c_x < end_x - margin) and (
                y + margin < c_y < end_y - margin
            )
            truths.append(hasRect)
            crops.append(image[y:end_y, x:end_x])
    crops = torch.stack(crops)
    print("shape of crops", crops.shape)
    return crops, truths


# -- Get some validation results ---------------------------------------------
model.to(device).eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    images, labels, coords = dataiter.next()
    print("images length", len(images))
    outputs = model(images)  # PREDICT
    loss = criterion(outputs.to(device), labels.to(device))
    print(loss)


# -- Plot windows and print labels -----------------------------------
n = 1
imgSample = images[n]
crops, truths = makeCrops(imgSample, 50, (100, 100), coords[n])

# -- Print x,y for one result -------
print("!-- FOR N = ", n)
print("y \n\t", [int(v) for v in labels[n]])
print("!-- y \n\t", [int(n) for n in truths])
print("!-- yhat \n\t", [int(round(o, 0)) for o in outputs[n].cpu().numpy()])
print("\n ------ x,y center + orient: ", coords[n], "\n")

# -- Print outputs for multiple results -------
for n in range(5):
    print("!-- y \t", [int(n) for n in labels[n]],
          "!-- yhat \t", [int(round(o, 0)) for o in outputs[n].cpu().numpy()],
          "\t loss: %.5f" % (criterion(outputs[n], labels[n].to(device))),)

# -- Show original image, and the sliding windo crops -------
plt.imshow(imgSample)

for (i, crop) in enumerate(crops):
    # print("1-index number of window: ", i+1, 'x', x, 'y', y, 'has rectangle?', hasRect)
    plt.figure()
    plt.suptitle("numero: %d" % (i))
    plt.imshow(crop)


# -- Main ---------------------------------------------
if __name__ == '__main__':
    main()
