from easydict import EasyDict as edict

__C = edict()

cfg = __C

# features about input dataset
__C.IN_CH = 1

__C.H = 512

__C.W = 512

__C.INIT_FEATURES = 32


### Training phase ###
__C.TRAIN = edict()

__C.TRAIN.BATCH_SIZE = 2

__C.TRAIN.LOG_PATH = "./logs"

__C.TRAIN.SAVE_PATH = "./saves"

__C.TRAIN.DEVICE = "cuda:0"

__C.TRAIN.EPOCH = 150

__C.TRAIN.SAVE_INTERVAL = 5

__C.TRAIN.EVAL_INTERVAL = 5

__C.TRAIN.LR_SCHEDULER = 70

__C.TRAIN.LR = 1


### Testing options ###
__C.TEST = edict()

__C.TEST.OUTPUT_PATH = "./outputs"


### Segmentation settings ###
__C.SEG = edict()

__C.SEG.OUT_CH = 4
# representation for segmentation, format of the output [segmentation]
__C.SEG.REP = ['DSegMap', 'VSegMap', 'BgSegMap', 'OtherSegMap']

assert len(__C.SEG.REP) == __C.SEG.OUT_CH
# path for segmentation folder
__C.SEG.FOLDER = 'C:/Research/LumbarSpine/OriginalData/RealSegmentationData/'

### Localization settings ###
__C.LOC = edict()

__C.LOC.NUM_CLASSES = 6

__C.LOC.ANCHORS = [[91.84, 49.6], [93.76, 39.36], [96.96, 56.64],
                   [84.8, 41.92], [89.28, 37.12], [90.24, 33.92],
                   [74.24, 54.4], [80.32, 58.56], [81.92, 45.44]]

__C.LOC.NUM_ANCHORS = 3

__C.LOC.NUM_CLASSES = 6

# [(116, 90), (156, 198), (373, 326),
#  (30, 61), (62, 45), (59, 119),
#  (10, 13), (16, 30), (33, 23)]

