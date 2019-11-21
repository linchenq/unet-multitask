from easydict import EasyDict as edict

__C = edict()

cfg = __C

# input channels [dataset]
__C.IN_CH = 1
# image height [dataset]
__C.H = 512
# image width [dataset]
__C.W = 512
# initial features [network]
__C.INIT_FEATURES = 32


# Training options
__C.TRAIN = edict()
# batch size for training phase
__C.TRAIN.BATCH_SIZE = 2
# log path
__C.TRAIN.LOG_PATH = "./logs"
# save path
__C.TRAIN.SAVE_PATH = "./saves"
# device
__C.TRAIN.DEVICE = "cuda:0"
# training epoches
__C.TRAIN.EPOCH = 202
# save interval
__C.TRAIN.SAVE_INTERVAL = 10
# evaluate interval
__C.TRAIN.EVAL_INTERVAL = 5
# decay step for scheduler
__C.TRAIN.LR_SCHEDULER = 100
# learning rate
__C.TRAIN.LR = 0.01


# Testing options
__C.TEST = edict()


# Segmentation settings
__C.SEG = edict()
## output channels for segmentation [segmentation]
__C.SEG.OUT_CH = 4
## representation for segmentation, format of the output [segmentation]
__C.SEG.REP = ['DSegMap', 'VSegMap', 'BgSegMap', 'OtherSegMap']
## assertion for the output channels [segmentation]
assert len(__C.SEG.REP) == __C.SEG.OUT_CH
## path for segmentation folder
__C.SEG.FOLDER = 'C:/Research/LumbarSpine/OriginalData/RealSegmentationData/'

# Localization settings
__C.LOC = edict()

__C.LOC.NUM_CLASSES = 6

__C.LOC.ANCHORS = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]

