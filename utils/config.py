# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.


opt = Config()