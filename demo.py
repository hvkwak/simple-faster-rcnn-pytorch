import os
import torch
import h5py
from utils.util import read_image
from models.faster_rcnn_vgg16 import FasterRCNNVGG16


## load image
img = read_image(os.path.dirname(os.path.abspath(__file__))+'/demo.jpg')
img = torch.from_numpy(img)[None]

## load pretrained model
filename = os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/converted.h5"
state_dict = h5py.File(filename, 'r')










state_dict.items()
for k, i in state_dict.items():
    print(k, i)

''' use this later:    
model.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model.named_parameters() if k in l})
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