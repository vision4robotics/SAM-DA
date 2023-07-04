# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "udatban_r50_l234"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64


__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.RESUME_D = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.BASE_LR_d = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

# __C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'COCO', 'GOT10K', 'NAT') # 'DET',  'LASOT'
__C.DATASET.SOURCE = ['VID', 'GOT10K'] # source domain
__C.DATASET.TARGET = ['NAT'] # target domain

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = './train_dataset/vid/crop511'
__C.DATASET.VID.ANNO = './train_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = -1

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = ''
__C.DATASET.YOUTUBEBB.ANNO = ''
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = 200000

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = ''
__C.DATASET.COCO.ANNO = ''
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = 100000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = ''
__C.DATASET.DET.ANNO = ''
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 200000

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = './train_dataset/got10k/crop511'
__C.DATASET.GOT10K.ANNO = './train_dataset/got10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 100
__C.DATASET.GOT10K.NUM_USE = -1

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = ''
__C.DATASET.LASOT.ANNO = ''
__C.DATASET.LASOT.FRAME_RANGE = 100
__C.DATASET.LASOT.NUM_USE = 200000

__C.DATASET.NAT = CN()
__C.DATASET.NAT.ROOT = './train_dataset/sam_nat/crop511'
__C.DATASET.NAT.ANNO ='./train_dataset/sam_nat/sam_nat_b.json'
__C.DATASET.NAT.ANNO_B ='./train_dataset/sam_nat/sam_nat_b.json'
__C.DATASET.NAT.ANNO_S ='./train_dataset/sam_nat/sam_nat_s.json'
__C.DATASET.NAT.ANNO_T ='./train_dataset/sam_nat/sam_nat_t.json'
__C.DATASET.NAT.ANNO_N ='./train_dataset/sam_nat/sam_nat_n.json'
__C.DATASET.NAT.FRAME_RANGE = 1
__C.DATASET.NAT.NUM_USE = -1 # 10000

__C.DATASET.VIDEOS_PER_EPOCH = 20000
__C.DATASET.VIDEOS_PER_EPOCH_B = 20000
__C.DATASET.VIDEOS_PER_EPOCH_S = 10000
__C.DATASET.VIDEOS_PER_EPOCH_T = 6667
__C.DATASET.VIDEOS_PER_EPOCH_N = 4000

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"
# ------------------------------------------------------------------------ #
# Align layer options
# ------------------------------------------------------------------------ #
__C.ALIGN = CN()

# Align layer
__C.ALIGN.ALIGN = False

__C.ALIGN.KWARGS = CN(new_allowed=True)

# ALIGN layer type
__C.ALIGN.TYPE = "Adjust_Transformer"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
__C.BAN = CN()

# Whether to use ban head
__C.BAN.BAN = False

# BAN type
__C.BAN.TYPE = 'MultiBAN'

__C.BAN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Point options
# ------------------------------------------------------------------------ #
__C.POINT = CN()

# Point stride
__C.POINT.STRIDE = 8

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamBANTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.14

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.45

# Interpolation learning rate
__C.TRACK.LR = 0.30

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.NAT = [0.473, 0.02, 0.385]

__C.HP_SEARCH.NAT_L = [0.466, 0.083, 0.404]

__C.HP_SEARCH.UAVDark70 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.UAVDark135 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.DarkTrack2021 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.NUT_L = [0.473, 0.02, 0.385]