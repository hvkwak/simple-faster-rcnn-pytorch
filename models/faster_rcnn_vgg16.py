from models.utils.config import opt
from models.faster_rcnn import FasterRCNN
from models.region_proposal_network import RegionProposalNetwork
# from models.roi_module import RoIPooling2D
from models.roi_module import RoI
import torch
from torchvision.models import vgg16
from torch import nn
from utils import array_tool as at
import torch.nn.functional as F
import torchvision.ops.roi_pool as roi_pool

def decom_vgg16():
    # loads vgg16 extractor(layers before FC) and 
    # classifier(layers of FC).

    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(torch.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)
    #
    print("-------------------VGG16 Model---------------------")
    print(model)
    print("---------------------------------------------------")
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier) # multiple positional arguments

    # freeze top4 conv - see if this is changed when it comes to training.
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    """
    *args means accepting the arbitrary numbers of positional arguments and
    **kwargs means accepting the arbitrary numbers of keyword arguments. 
    In here, *args, **kwargs are called packing.
    """
    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16
    # we are dealing with face detection
    # class 0: non face(background)
    # class 1: face
    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):

        extractor, classifier = decom_vgg16()
        rpn = RegionProposalNetwork(in_channels=512, 
                                    mid_channels=512, 
                                    ratios=[0.5, 1, 2], 
                                    anchor_scales=[8 ,16, 32], 
                                    feat_stride=16, 
                                    proposal_creator_params=dict())
        head = VGG16RoIHead(
            # check if n_fg_class is correct
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super().__init__(extractor, rpn, head)



class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super().__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        # check how RoIPooling works and implement it.
        # self.roi = RoI
        self.roi = roi_pool
        # self.roi = RoIPooling2D(self.roi_size, self.roi_size)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()

        #####################################################################
        # X, Y changes???                                                   #
        #####################################################################
        # combine rois and roi_indices
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        # indices_and_rois = (index, (x1, y1, x2, y2))
        # where x in [0, 600], y in [0, 800]
        # xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # indices_and_rois =  xy_indices_and_rois.contiguous()
        #####################################################################
        
        # roi pooling:
        # pool = self.roi.apply(x, indices_and_rois)
        pool = self.roi(x, indices_and_rois[:, [0, 2, 1, 4, 3]], (7, 7), 1/16)
        # pool = self.roi(x, indices_and_rois, (7, 7), 1/16)

        pool = pool.view(pool.size(0), -1) # flatten
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores




def normal_init(m, mean, stddev, truncated=False):
    # initialize the layer random normal
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()