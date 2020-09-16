import os
import sys
from utils.config import opt
import numpy as np
import torch
import h5py
from utils.util import read_image
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.array_tool import rename
from utils.vis_tool import vis_bbox
from models.classifier import Classifier_VGG
import cv2 as cv

# from utils import array_tool as at
# from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from PIL import Image

## load image
image_name = "/demo5.png"
image_name = os.path.dirname(os.path.abspath(__file__))+image_name
img = read_image(image_name)
img = torch.from_numpy(img)[None]

## FASTER RCNN
faster_rcnn = FasterRCNNVGG16()
faster_rcnn.demo_image = image_name

## Classifier
vgg_descriptors_path = os.getcwd() + "/facerecognition/FaceRecClient/files/"
classifier = Classifier_VGG(vgg_descriptors_path)
classifier.load_model()

## load pretrained model
## this pretrained model is available at:
## https://github.com/playerkk/face-py-faster-rcnn
filename = '/home/hyobin/Documents/in-facedemo/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/vgg16_faster_rcnn_iter_80000.caffemodel.pt'
state_dict = torch.load(filename)
state_dict = {k: v for k, v in state_dict.items()}
state_dict = rename(faster_rcnn.named_parameters(), state_dict)
opt.caffe_pretrain=True

## loading weights of FASTER RCNN
faster_rcnn.load_state_dict({l : torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in faster_rcnn.named_parameters() if k in l})

## FASTER RCNN feed forward
bboxes, labels, scores = faster_rcnn.predict(img, visualize = True)

## classifier.predict()
predicted_distances, predicted_names = classifier.predict(img, bboxes)

## visualize
img1 = Image.open(image_name)
vis_bbox(img1, bboxes, scores, predicted_names, predicted_distances)
