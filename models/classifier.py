import numpy as np
import sys
import os
import torch
import cv2 as cv
from os import listdir
from models.vggmodel import VGG
import torchvision
from PIL import Image
import re

class Classifier_VGG(object):

    def __init__(self, file_path):
        # file_path = ".../facerecognition/FaceRecClient/files/"
        self.file_path = file_path
        self.descriptor_path = self.file_path+"vgg_files/descriptors/"
        self.weights_path  = self.file_path+"vgg_files/vgg_face_torch/VGG_FACE.t7"
        self.train_path = file_path+"images/Train/"
        self.labels, self.names = self.labels_creator(self.train_path)
        '''
        # Startwert: Infinity
        self.best_distance = np.Inf
        self.best_label = n+1
        self.best_index = np.Inf
        self.best_class = np.Inf

        self.threshold = 3000
        '''
    @torch.no_grad()
    def load_model(self):
        
        # model ready to predict:
        self.model = VGG()
        self.model.load_weights(self.weights_path)
        self.model.eval()

        # get descriptors ready
        self.num_class = len(listdir(self.descriptor_path))
        descriptors = [None]*self.num_class
        for i in range(self.num_class):
            descriptor = torch.load(self.descriptor_path+"Train_descriptors{}.pt".format(i+1))
            # Zentroid berechnen:
            descriptors[i] = torch.mean(descriptor, axis = 0)
        # save the zentroids
        self.descriptors = torch.stack(descriptors)
    
    @torch.no_grad()
    def predict(self, img, bboxes):
        # img = self.read_PIL_image(img_path)
        # resize_image = torchvision.transforms.Resize((224, 224))
        '''
        # change the PIL Image to OpenCV Image first: RGB -> BGR
        img = np.asarray(img)
        img = img[:, ::-1, :, :].copy()
        '''
        bbox_num = bboxes[0].shape[0]
        predicted_distances = [None]*bbox_num
        predicted_names = [None]*bbox_num
        for i in range(bbox_num):
            y1, x1, y2, x2 = bboxes[0][i, :].astype(np.int32)
            y1, x1 = y1+1, x1+1
            img_piece = np.asarray(img[0]).transpose(1, 2, 0)
            img_piece = cv.resize(img_piece[y1:y2, x1:x2], (224, 224))
            img_piece = torch.Tensor(img_piece).permute(2, 0, 1).view(1, 3, 224, 224)
            img_piece -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
            img_piece = self.model(img_piece)[0]

            # euclidean distance to zentroids
            NN = torch.sum((self.descriptors - img_piece[0])**2, 1)
            buffer = NN.min(0)
            predicted_distances[i] = buffer[0]
            predicted_names[i] = self.names[buffer[1]]
        return(predicted_distances, predicted_names)

    def labels_creator(self, train_path):
        folders = listdir(train_path) # list
        labels = list(range(1, len(folders)+1))
        names = [None]*len(listdir(train_path))
        r = re.compile(r"[^a-zA-Z]")
        for i in range(len(folders)):            
            names[int(folders[i][0:2])-1] = r.sub("", folders[i])    
        return labels, names