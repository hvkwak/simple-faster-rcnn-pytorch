import os
import sys
sys.path.append(os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/")

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from models.utils.nms import non_maximum_suppression
from models.utils.bbox_tools import loc2bbox
from utils.array_tool import tonumpy, totensor
from data.dataset import preprocess
from utils.util import read_image

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.
    """

    def __init__(self, extractor, rpn, head, 
                    loc_normalize_mean = (0., 0., 0., 0.),
                    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        # in Python3, inheritance and initialize like this:
        super().__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')        

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class
  

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]
        h = self.extractor(x)

        # rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        # rpn_locs, rpn_scores, anchors are obsolete
        _, _, rois, roi_indices, _ = self.rpn(h, img_size, scale)

        # visualize RPN results to see if they are working correctly:
        visualize_RPN(rois, self.scale)

        # feed forward weiter:
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        # non maximum suppresion before final predictions
        bbox = list()
        label = list()
        score = list()

        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):

            # take class l figures and their estimated probabilities
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]

            # check if they exceed the threshold 
            # take those who exceeded
            # mask = prob_l > 0.7
            mask = prob_l.argsort()[::-1][:3] # take top 10
            cls_bbox_l = cls_bbox_l[mask, :]
            prob_l = prob_l[mask]
            
            '''
            # indexes to keep
            keep = non_maximum_suppression(np.array(cls_bbox_l), 0.5)
            print(keep[1], " out of ", 70, " still there")
            # keep = non_maximum_suppression(np.array(cls_bbox_l), self.nms_thresh)
            try:
                keep = keep[0][:np.where(keep[0][1:] == 0)[0].min()]
                keep = keep.numpy()
            except:
                keep = keep[0]
            
            
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(mask),)))
            score.append(prob_l[keep])
            '''
            bbox.append(cls_bbox_l)
            score.append(prob_l)
            label.append((l - 1) * np.ones((len(mask),)))

        return bbox, label, score

    @torch.no_grad()
    def predict(self, imgs, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize') # Visualize mode
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img, scale = preprocess(tonumpy(img))
                self.scale = scale
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        
        # create output lists
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            # change it to tensor
            # [None] addes up one more dimension
            img = totensor(img[None]).float()

            # scale factor
            scale = img.shape[3] / size[1]

            # fast forward the image
            # img   -> (extractor+rpn+head) -> roi_cls_loc, roi_scores, rois
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)

            # NOTE:
            # rois.shape =  (300, 4)
            # where 4 corresponds to (y1, x1, y2, x2)
            # x in [0, 600], y in [0, 800]

            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data

            # change rois to tensor
            roi = totensor(rois) / scale

            # check the codes below.
            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = torch.Tensor(self.loc_normalize_mean). \
                repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),
                                tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            # change the form (N, 4) 
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)

            # clamp in range of [0, size[0]]
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = tonumpy(F.softmax(totensor(roi_score), dim=1))

            # change tensors to numpy
            raw_cls_bbox = tonumpy(cls_bbox)
            raw_prob = tonumpy(prob)

            # non maximum suppression
            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train() # change it back to train mode.
        return bboxes, labels, scores
        

def visualize_RPN(rois, scale):
    # Visualize RPN results
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    ## load image
    image_name = "/demo.jpg"
    img1 = Image.open('/home/hyobin/Documents/in-facedemo/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/'+image_name)
    # img1 = read_image(os.path.dirname(os.path.abspath(__file__))+'/demo.jpg')
    fig, ax = plt.subplots(1)
    ax.imshow(img1)

    # visualize top 5 images
    for i in range(5):
        y1, x1, y2, x2 = rois[i, :]
        y1, x1, y2, x2 = y1/scale, x1/scale, y2/scale, x2/scale
        h = y2 - y1
        w = x2 - x1
        rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()