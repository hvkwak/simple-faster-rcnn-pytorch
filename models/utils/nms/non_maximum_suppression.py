import numpy as np
# import cupy as cp
import torch


def non_maximum_suppression(bboxes, threshold=0.7):
    '''
    Non Maximum Suppresion courtesy of:
    https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185#file-nms_pytorch-py-L34
    '''
    '''
    Args:
        bboxes: (tensor) The location predictions for the img, [num_priors, 4]
                This is already sorted according to scores from highest 
                to lowest.
        threshold: IoU threshold
    '''
    # change bboxes from np.ndarray to tensor:
    bboxes = torch.from_numpy(bboxes)

    # return zeros if it is empty
    n_bboxes = bboxes.shape[0]
    idx = torch.arange(n_bboxes)
    keep = torch.zeros(n_bboxes).long()

    if not bboxes.shape[0] > 0:
        return keep
    
    y1 = bboxes[:, 0]
    x1 = bboxes[:, 1]
    y2 = bboxes[:, 2]
    x2 = bboxes[:, 3]

    area = torch.mul(x2 - x1, y2 - y1) # IoU / area

    yy1 = bboxes.new()
    xx1 = bboxes.new()
    yy2 = bboxes.new()
    xx2 = bboxes.new()
    w = bboxes.new()
    h = bboxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[0] # index of current largest val
        keep[count] = i
        count = count+1

        # if there's only one element, break
        if idx.size(0) == 1:
            break

        # take the first element out
        idx = idx[1:]
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        # we take care of just dim using resize_as_()
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        # then compute w and h:
        w = xx2 - xx1
        h = yy2 - yy1
        
        # make sure that there are no values smaller than 0.
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(threshold)]
    return keep, count

#try:
#    from _nms_gpu_post import _nms_gpu_post
#except:
#    import warnings
#    warnings.warn('''
#    the python code for non_maximum_suppression is about 2x slow
#    It is strongly recommended to build cython code: 
#    `cd model/utils/nms/; python3 build.py build_ext --inplace''')
#    from _nms_gpu_post_py import _nms_gpu_post
'''
def non_maximum_suppression(bbox, thresh, score=None,
                            limit=None):
    """Suppress bounding boxes according to their IoUs.

    This method checks each bounding box sequentially and selects the bounding
    box if the Intersection over Unions (IoUs) between the bounding box and the
    previously selected bounding boxes is less than :obj:`thresh`. This method
    is mainly used as postprocessing of object detection.
    The bounding boxes are selected from ones with higher scores.
    If :obj:`score` is not provided as an argument, the bounding box
    is ordered by its index in ascending order.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    :obj:`score` is a float array of shape :math:`(R,)`. Each score indicates
    confidence of prediction.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    an input. Please note that both :obj:`bbox` and :obj:`score` need to be
    the same type.
    The type of the output is the same as the input.

    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.

    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    """

    return _non_maximum_suppression_gpu(bbox, thresh, score, limit)


def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    #
    # top score bboxes, but still too many.

    # returns an empty list if it is empty.
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    # if score is not None,
    # sort it from highest to lowest.
    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = np.arange(n_bbox, dtype=np.int32)

    sorted_bbox = bbox[order, :]
    selec, n_selec = _call_nms_kernel(
        sorted_bbox, thresh)
    selec = selec[:n_selec]
    selec = order[selec]
    if limit is not None:
        selec = selec[:limit]
    # return np.asnumpy(selec)
    return selec


def _call_nms_kernel(bbox, thresh):
    # PyTorch does not support unsigned long Tensor.
    # Doesn't matter,since it returns ndarray finally.
    # So I'll keep it unmodified.

    # n_bbox = the number of bboxes
    n_bbox = bbox.shape[0]
    
    # per block it is 64
    threads_per_block = 64

    # make several blocks of bboxes, 64 bboxes per thread.
    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)

    # tuples in python are ordered and unchangeable
    blocks = (col_blocks, col_blocks, 1)
    threads = (threads_per_block, 1, 1)
    # (e.g.) if n_bbox = 3000, threads_per_block = 64 then:
    # col_blocks = 47
    # blocks = (47, 47, 1)
    # threads = (64, 1, 1)
    
    mask_dev = np.zeros((n_bbox * col_blocks,), dtype=np.uint64)
    bbox = np.ascontiguousarray(bbox, dtype=np.float32)
    kern = _load_kernel('nms_kernel', _nms_gpu_code)
    kern(blocks, threads, args=(np.int32(n_bbox), np.float32(thresh),
                                bbox, mask_dev))

    mask_host = mask_dev.get()
    selection, n_selec = _nms_gpu_post(
        mask_host, n_bbox, threads_per_block, col_blocks)
    return selection, n_selec

def _load_kernel(kernel_name, code, options=()):
    cp.cuda.runtime.free(0)
    assert isinstance(options, tuple)
    kernel_code = cp.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)
'''