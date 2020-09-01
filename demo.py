import os
import sys
sys.path.append(os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/")
from utils.config import opt
import numpy as np
import torch
import h5py
from utils.util import read_image
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.array_tool import rename
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from data.voc_dataset import VOC_BBOX_LABEL_NAMES

## load image
image_name = "/demo.jpg"
img = read_image(os.path.dirname(os.path.abspath(__file__))+image_name)
img = torch.from_numpy(img)[None]

## model
faster_rcnn = FasterRCNNVGG16()

## load pretrained model
## this pretrained model is available at:
## https://github.com/playerkk/face-py-faster-rcnn
# try 1
filename = '/home/hyobin/Documents/in-facedemo/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/chainer_best_model_converted_to_pytorch_0.7053.pth'
state_dict = torch.load(filename)
faster_rcnn.load_state_dict(state_dict)
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model


'''
# try 2
filename = "'/home/hyobin/Documents/in-facedemo/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/converted.h5"
state_dict = h5py.File(filename, 'r')
state_dict = {k: v for k, v in state_dict.items()}
state_dict = rename(faster_rcnn.named_parameters(), state_dict)

## load weights
faster_rcnn.load_state_dict({l : torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in faster_rcnn.named_parameters() if k in l})
'''

## predict
bboxes, labels, scores = faster_rcnn.predict(img, visualize = True)

# visualize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

img1 = Image.open(os.path.dirname(os.path.abspath(__file__))+image_name)
# img1 = read_image(os.path.dirname(os.path.abspath(__file__))+'/demo.jpg')
fig, ax = plt.subplots(1)
ax.imshow(img1)
for i in range(bboxes[0].shape[0]):
    y1, x1, y2, x2 = bboxes[0][i, :]
    label = labels[0][i]
    score = np.round(scores[0][i], 3)
    h = y2 - y1
    w = x2 - x1
    rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.text(x1, y1, VOC_BBOX_LABEL_NAMES[label]+"("+str(score)+")")
    ax.add_patch(rect)
plt.show()












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