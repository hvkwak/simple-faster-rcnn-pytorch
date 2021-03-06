import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import cv2 as cv
import numpy as np
import os
# import tqdm


class VGG(nn.Module):
    ''' VGG model for classification
    '''

    def __init__(self):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]

        # 3 input image channel, 64 output channels, 3x3 square convolution
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1)

        # 64 feature maps again to 64 feature maps
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding = 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding = 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding = 1)

        self.FC6 = nn.Linear(512 * 7 * 7, 4096) # 7 * 7 from image dimension
        self.FC7 = nn.Linear(4096, 4096)
        self.FC8 = nn.Linear(4096, 2622) # 2622 classes

    def forward(self, x):

        # input x.dim = (224, 224, 3)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, (2, 2)) # max pooling, window size of (2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, (2, 2))        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, (2, 2))

        # flatten the feature maps: (1, N), where -1 does the job to calculate N
        x = x.view(x.size(0), -1)

        # FCs
        x = F.relu(self.FC6(x))
        x = F.dropout(x, 0.5, self.training)
        x7 = F.relu(self.FC7(x))
    
        x8 = F.dropout(x7, 0.5, self.training)
        return(x7, self.FC8(x8))

    def load_weights(self, path):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "FC%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
