## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2)
        self.bn32 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.bn128 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,5)
        self.conv5 = nn.Conv2d(256,512,5)
        self.bn512 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512,136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))  # 3*224*224 -> 32*110*110
        x = self.bn32(x)
        x = self.pool(F.relu(self.conv2(x)))  # 32*110*110 ->64*53*53
        x = self.pool(F.relu(self.conv3(x)))  # 64*53*53 -> 128*24*24
        x = self.bn128(x)
        x = self.pool(F.relu(self.conv4(x)))  # 128*24*24 -> 256*10*10
        x = F.relu(self.conv5(x))                  # 256*10*10 -> 512*6*6
        x = self.bn512(x)
        # print(x.size())
        x = self.avgpool(x)  # 512*1*1
        x = x.view(x.size(0),-1)
        # print(x.size())
        x = self.fc1(x)     # 136

        # a modified x, having gone through all the layers of your model, should be returned
        return x
