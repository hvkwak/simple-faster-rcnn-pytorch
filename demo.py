import sys
import os
sys.path.append(os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/")
import torch
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

## load image
image_name = "/demo.jpg"
img = read_image(os.path.dirname(os.path.abspath(__file__))+image_name)
img = torch.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()

# try 1
filename = os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/chainer_best_model_converted_to_pytorch_0.7053.pth"
state_dict = torch.load(filename)
faster_rcnn.load_state_dict(state_dict)

opt.caffe_pretrain=True # this model was trained from caffe-pretrained model

_bboxes, _labels, _scores = faster_rcnn.predict(img,visualize=True)

vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))