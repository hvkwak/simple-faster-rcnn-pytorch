import os
import torch
from utils.util import read_image
from models.faster_rcnn_vgg16 import FasterRCNNVGG16


## load image
img = read_image(os.path.dirname(os.path.abspath(__file__))+'/demo.jpg')
img = torch.from_numpy(img)[None]

## load pretrained model
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