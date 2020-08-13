import torch
import numpy as np
from torch.autograd import Function


class RoI(Function):

    '''
    def __init__(self, outh, outw):
        self.outh, self.outw = outh, outw
    '''

    @staticmethod
    def forward(ctx, input, rois, roi_size = (7, 7)):
        '''
        :param input: feature_map, (1, C, H, W)
        :param rois: (1, N, 4) N refers to bbox num, 4 represent (ltx, lty, w, h) 
        '''
        feature_map = input.contiguous() # not sure if .contiguous() is needed
        rois = rois.contiguous()
        in_size = B, C, H, W = feature_map.size()
        # the number of roi
        N = rois.size(0)
        outw, outh = roi_size # 7, 7

        # normally it is the form of (1, 512, 7, 7)
        output = torch.zeros(N, C, outw, outh)
        indices_memory = torch.zeros(N, outh, outw, 4+C).int()

        # roi pooling: 7 x 7
        for i in range(N):
            roi = rois[0][i]
            x, y, w, h = roi

            current_roi = feature_map[:, :, y:y+h, x:x+w].contiguous()
            # take min{w, h} and upscale until it is bigger than outw or outh
            if w//outw == 0 or h//outh == 0:
                min_d = np.min((w, h))
                multiplier = 1
                while min_d*multiplier < 7:
                    multiplier = multiplier * 2
                m = torch.nn.Upsample(scale_factor=multiplier, mode = 'nearest')
                current_roi = m(current_roi[:, :, y:y+h, x:x+w])

            bB, bC, bH, bW = current_roi.size()

            # make sure that these are integers
            dh = bH//outh
            dw = bW//outw
            
            for j in range(outw):
                for k in range(outh):
                    # .contiguous() needed?
                    buffer = current_roi[:, :, y+(dh*k):y+(dh*(k+1)), x+(dw*j):x+(dw*(j+1))].contiguous()
                    output[i, :, k, j] = buffer.view(buffer.size(1), -1).max(dim = -1).values
                    y1, x1, y2, x2 = y+(dh*k), x+(dw*j), y+(dh*(k+1)), x+(dw*(j+1))
                    indices = buffer.view(buffer.size(1), -1).argmax(dim = -1)
                    indices_memory[i, k, j, :] = torch.cat(torch.Tensor([y1, x1, y2, x2], indices))                    
        ctx.save_for_backward(input, rois, indices_memory, roi_size)
        return output
        
    '''
    Not sure if this backward works
    nor sure if this backward is needed for just feed-forward demo.
    '''
    @staticmethod
    def backward(ctx, grad_output):
        input, rois, ind, roi_size = ctx.saved_tensors
        
        output = torch.zeros(input.size())

        for i in range(rois.size(0)):
            for j in range(roi_size[0]):
                for k in range(roi_size[1]):
                    output[i, :, ind[0]:ind[2], ind[1]:ind[3]][ind[4:]] = grad_output[i, j, k, :]        
        return output

class RoIPooling2D(torch.nn.Module):
    # def __init__(self, outh, outw, spatial_scale):
    def __init__(self, outh, outw):
        # super(RoIPooling2D, self).__init__()
        super().__init__()
        # self.RoI = RoI(outh, outw, spatial_scale)
        self.RoI = RoI(outh, outw)

    def forward(self, x, rois):

        return self.RoI(x, rois)