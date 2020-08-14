import os
import sys
sys.path.append(os.getcwd() + "/facerecognition/PyFaceRecClient/simple-faster-rcnn-pytorch/")

import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn
from models.utils.bbox_tools import generate_anchor_base, bbox_iou
from models.utils.creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """


    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super().__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride

        # this rpn is the parent model
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0] # number of anchors

        # nn.Conv2(in_channels, out_channels, kernel_size, stride, padding)
        # feature maps dimension of (1, 512, 50, 50)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # For example: how are the dimensions? 
        # 1. Dim of feature maps after self.conv1 
        # : (1, 512, hh, ww)
        #
        # # self.loc
        # 2. Dim after (1 x 1) Convolution with n_anchor*4 filters # self.loc
        # : (1, n_anchor*4, hh, ww)
        # this will later be then permuted to be (1, hh, ww, n_anchor*4)
        # 
        # # self.score
        # 3. Dim after (1 x 1) Convolution with n_anchor*2 filters 
        # : (1, n_anchor*2, hh, ww)

        # normal initilaization? no loading pretrained model?
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape # input image x

        # anchors at all pixels generated here (hh * ww * n_anchor, 4)
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        # number of anchors at a pixel (here: 9)
        n_anchor = anchor.shape[0] // (hh * ww)

        h = F.relu(self.conv1(x))

        # (1, n_anchor*4, hh, ww) coordinates
        # self.loc is just a linear function!
        rpn_locs = self.loc(h)

        # permute() returns non-"contiguous" tensor.    
        # change it to a contiguous() (= individual) tensor.
        # narrow(), view(), expand() and transpose()

        # permute() this so that this will be then: (1, hh, ww, n_anchor*4)
        # view(n, -1, 4) flattens it into (1, hh*ww*n_anchor, 4)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # self.score then generates (1, n_anchor*2, hh, ww)
        # two values at each point of the feature map.
        rpn_scores = self.score(h)
        
        # change it to (1, hh, ww, n_anchor*2), and make it contiguous
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()

        # dim = 4 indicates that we'd like to have softmax function values
        # at [n, hh, ww, n_anchor, 2][4], so the last "2".
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)

        # takes the scores of the object being a figure
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()

        # flatten the figure scores. -1 is the "der Positionshalter", 
        # which does the rest of the work for you simply.
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        
        # flatten the original scores so the dim is then like....
        # e.g.: (1, 22500, 2)
        rpn_scores = rpn_scores.view(n, -1, 2)
        # difference? fg_scores and scores?

        # n is batch_size
        rois = list()
        roi_indices = list()
        for i in range(n):
            # self.proposal_layer.__call__(...)
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            # number all the rois from one image in the batch
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            # append them, this can slow down the speed.
            rois.append(roi)
            roi_indices.append(batch_index)
        # all the rois concatenate
        rois = np.concatenate(rois, axis=0)

        # all the batch indexes concatenate
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor
        


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # m is a layer of nn
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    '''
    Inputs:
    anchor_base : anchor base generated from (0, 0) of the feature map(s)
    feat_stride : down sample scale. (here : 16)
    height, width: height and width of feature maps

    Outputs:
    an anchorbase generated at all the pixels of feature maps, which is then
    transformed into the scale of original image: ex) 800 x 800
    '''
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
