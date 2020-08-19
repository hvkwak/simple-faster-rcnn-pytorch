import os
import sys
sys.path.append(os.getcwd() + "/facerecognition/PyFaceRecClient/FASTER_RCNN/")
import numpy as np
import torch
import h5py
from utils.util import read_image
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.array_tool import rename

## load image
img = read_image(os.path.dirname(os.path.abspath(__file__))+'/demo.jpg')
img = torch.from_numpy(img)[None]

## model
faster_rcnn = FasterRCNNVGG16()
# trainer = FasterRCNNTrainer(faster_rcnn)

## load pretrained model
## this pretrained model is available at:
## https://github.com/playerkk/face-py-faster-rcnn
filename = os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/converted.h5"
state_dict = h5py.File(filename, 'r')

state_dict = {k: v for k, v in state_dict.items()}
state_dict = rename(faster_rcnn.named_parameters(), state_dict)

## load weights
faster_rcnn.load_state_dict({l : torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in faster_rcnn.named_parameters() if k in l})

## predict
print(faster_rcnn.predict(img, visualize = True))













''' use this later:

'''

'''
path = '/home/hyobin/Documents/FASTRCNN_demo/chainer_best_model_converted_to_pytorch_0.7053.pth'
pretrained_dict = torch.load(path)

## create fasterRCNNVGG16
faster_rcnn_extractor = FasterRCNNVGG16()

## load extractor and load its dict
faster_rcnn_extractor_dict = faster_rcnn_extractor.state_dict()

## update its dict to pretrained_dict
## make sure that there are no irrelevant values
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in faster_rcnn_extractor_dict}
faster_rcnn_extractor_dict.update(pretrained_dict) # update
faster_rcnn_extractor.load_state_dict(faster_rcnn_extractor_dict) 

print(faster_rcnn_extractor(img))
'''