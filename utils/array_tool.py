"""
tools to convert specified type
"""
import torch as t
import numpy as np


def rename(named_parameters, state_dict):
    '''
    renames the parameter of pretrained model

    Input:
    name : the name of parameter from pretrained model
    named_parameters: parameters from faster_rcnn model
    '''
    conv_num = np.array([0, 0, 2, 2,\
                        5, 5, 7, 7, \
                        10, 10, 12, 12, 14, 14,\
                        17, 17, 19, 19, 21, 21,\
                        24, 24, 26, 26, 28, 28])
    conv_layer = np.array(["1_1", "1_1", "1_2", "1_2",\
                           "2_1", "2_1", "2_2", "2_2",\
                           "3_1", "3_1", "3_2", "3_2", "3_3", "3_3",\
                           "4_1", "4_1", "4_2", "4_2", "4_3", "4_3",\
                           "5_1", "5_1", "5_2", "5_2", "5_3", "5_3"])
    for k in list(state_dict.keys()):
        if k.startswith("conv"):
            if k.endswith("bias"):
                new_name = "extractor."+str(conv_num[conv_layer == k[4:7]][0])+".bias"
                state_dict[new_name] = state_dict.pop(k)
            else: # weight
                new_name = "extractor."+str(conv_num[conv_layer == k[4:7]][0])+".weight"
                state_dict[new_name] = state_dict.pop(k)
        # VGG Head
        elif k == "fc6.bias":
            state_dict["head.classifier.0.bias"] = state_dict.pop(k)
        elif k == "fc6.weight":
            state_dict["head.classifier.0.weight"] = state_dict.pop(k)
        elif k == "fc7.bias":
            state_dict["head.classifier.2.bias"] = state_dict.pop(k)
        elif k == "fc7.weight":
            state_dict["head.classifier.2.weight"] = state_dict.pop(k)
        elif k == "bbox_pred.bias":
            state_dict["head.cls_loc.bias"] = state_dict.pop(k)
        elif k == "bbox_pred.weight":
            state_dict["head.cls_loc.weight"] = state_dict.pop(k)
        elif k == "cls_score.bias":
            state_dict["head.score.bias"] = state_dict.pop(k)
        elif k == "cls_score.weight":
            state_dict["head.score.weight"] = state_dict.pop(k)
        # RPN
        elif k == "rpn_bbox_pred.bias": # rpn_bbox_pred
            state_dict["rpn.loc.bias"] = state_dict.pop(k)
        elif k == "rpn_bbox_pred.weight":
            state_dict["rpn.loc.weight"] = state_dict.pop(k)
        elif k == "rpn_cls_score.weight": # rpn_cls_score
            state_dict["rpn.score.weight"] = state_dict.pop(k)
        elif k == "rpn_cls_score.bias":
            state_dict["rpn.score.bias"] = state_dict.pop(k)
        elif k == "rpn_conv":
            state_dict["rpn.conv1.weight"] = np.array(state_dict['rpn_conv']['3x3.weight'])
            state_dict["rpn.conv1.bias"] = np.array(state_dict['rpn_conv']['3x3.bias'])
            del state_dict["rpn_conv"]
        else: # something was not right.
            return ValueError
    return(state_dict)


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=False): # default changed to cuda=False
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()