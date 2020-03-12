from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1.Data General Setting. Can be replaced in respective sets
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.DATA = CN()
# -----------------------------------------------------------------------------
# Data.Dataset
# -----------------------------------------------------------------------------
_C.DATA.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATA.DATASETS.NAMES = ('dukemtmc')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATA.DATASETS.ROOT_DIR = ('./data')
# -----------------------------------------------------------------------------
# Data.DataLoader
# -----------------------------------------------------------------------------
_C.DATA.DATALOADER = CN()
# Number of data loading threads
_C.DATA.DATALOADER.NUM_WORKERS = 4
# Sampler for data loading
_C.DATA.DATALOADER.SAMPLER = 'softmax_rank'
# Number of images per batch during test
_C.DATA.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per batch during test
_C.DATA.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4

_C.DATA.DATALOADER.IMS_PER_BATCH = _C.DATA.DATALOADER.CATEGORIES_PER_BATCH * _C.DATA.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
# -----------------------------------------------------------------------------
# Data.TRANSFORM
# -----------------------------------------------------------------------------
_C.DATA.TRANSFORM = CN()
# Size of the image during training
_C.DATA.TRANSFORM.SIZE = [384, 128]
# 掩膜缩放比例
_C.DATA.TRANSFORM.MASK_SIZE_RATIO = 4
# Values to be used for image normalization
_C.DATA.TRANSFORM.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.DATA.TRANSFORM.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.DATA.TRANSFORM.PADDING = 10



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.Model General Setting. Can be replaced in respective sets  Structure Information
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
###  se_resnext50 , se_resnet50 , resnet50 ,resnet34, mobilenetv3,resnet50_ibn_a,resnet50_ibn_a_old
_C.MODEL.NAME = 'multi_bagnet'
# If block is pre_activated, options: 1 or 0
_C.MODEL.PRE_ACTIVATION = 1
# how block output fuse, options: "concat", "add", "none"
_C.MODEL.FUSION_TYPE = "concat"
# baselineOutputType 和 classifierType  "f-c" "pl-c" "fl-n"
_C.MODEL.BASE_CLASSIFIER_COMBINE_TYPE = "f-c"


# supervisedType 3个支路的调配方案  若改变该项，则下述选项设定将无效    #"weak-reload", "semi", "self-semi", "none"
# "none", "weakSu-segRe", "strongSu-segRe", "strongSu-gcamRe", "gcamRe"
_C.MODEL.BRANCH_CONFIG_TYPE = "none"
# num of samples used in branch
_C.MODEL.BRANCH_IMG_NUM = 0

# 1.Segmnetation Bracnch
# hookType   "featureReserve":保存transition层features, "rflogitGenerate":生成rf_logit_map, "none"
_C.MODEL.HOOK_TYPE = "none"
# segType "denseFC", "bagFeature", "none"， "gradCAM"
_C.MODEL.SEGMENTATION_TYPE = "none"
# seg output channels
_C.MODEL.SEG_NUM_CLASSES = 1
# seg supervised type   # "none", "strong", "weak", "joint"
_C.MODEL.SEG_SUPERVISED_TYPE = "strong"

# 2.Grad-CAM Branch
# Grad-CAM 的作用限制
_C.MODEL.GRAD_CAM_TYPE = "none"  # "none", "supervise_seg", "reload"

# 3.Reload Branch
# masked img reload type  "none", "seg_mask", ""gradcam_mask""  （input）
_C.MODEL.MASKED_IMG_RELOAD_TYPE = "none"








# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3.LOSS General Setting. Can be replaced in respective sets  Structure Information
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#LOSS configuration
_C.LOSS = CN()
# The loss type of metric loss
# options:'cranked_loss','ranked_loss',
_C.LOSS.TYPE = 'ranked_loss'
# Margin of ranked list loss
_C.LOSS.MARGIN_RANK = 1.3  ### R = ALPHA - MARGIN_RANK
_C.LOSS.ALPHA = 2.0
_C.LOSS.TVAL = 1.0
_C.LOSS.WEIGHT = 0.4       ### loss = softmax + w*ranked_loss


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4.Solver
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# Path to checkpoint and saved log of trained model
_C.SOLVER.OUTPUT_DIR = "work_space"
# -----------------------------------------------------------------------------
# OPTIMIZER configuration
# -----------------------------------------------------------------------------
_C.SOLVER.OPTIMIZER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER.NAME = "Adam"
# Momentum
_C.SOLVER.OPTIMIZER.MOMENTUM = 0.9
# Settings of weight decay
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0005
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS = 0.
# -----------------------------------------------------------------------------
# SCHEDULER configuration
# -----------------------------------------------------------------------------
_C.SOLVER.SCHEDULER = CN()
# Base learning rate
_C.SOLVER.SCHEDULER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.SCHEDULER.BIAS_LR_FACTOR = 2
# decay rate of learning rate
_C.SOLVER.SCHEDULER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.SCHEDULER.STEPS = (30, 55)
# warm up factor
_C.SOLVER.SCHEDULER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.SCHEDULER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.SCHEDULER.WARMUP_METHOD = "linear"
# center loss lr
_C.SOLVER.SCHEDULER.LOSS_LR = 0.05
# if train from the head
_C.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD = 1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specific Setting - Train Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# If test with re-ranking, options: 'yes','no'
_C.TRAIN.RE_RANKING = 'no'
# Path to trained model
_C.TRAIN.WEIGHT = ""
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TRAIN.FEAT_NORM = 'yes'
# -----------------------------------------------------------------------------
# Train Transform
# -----------------------------------------------------------------------------
_C.TRAIN.TRANSFORM = CN()
# Random probability for image horizontal flip
_C.TRAIN.TRANSFORM.PROB = 0.5
# Random probability for random erasing
_C.TRAIN.TRANSFORM.RE_PROB = 0.5
# -----------------------------------------------------------------------------
# Train Dataloader
# -----------------------------------------------------------------------------
_C.TRAIN.DATALOADER = CN()
# Number of categories per batch
_C.TRAIN.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per category in a batch
_C.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of images per batch
_C.TRAIN.DATALOADER.IMS_PER_BATCH = _C.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * _C.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
# Number of Mask per batch
_C.TRAIN.DATALOADER.MASK_PER_BATCH = 1
# accumulation_steps (only used in train)
_C.TRAIN.DATALOADER.ACCUMULATION_STEP = 1


# -----------------------------------------------------------------------------
# Train Trick
# -----------------------------------------------------------------------------
_C.TRAIN.TRICK = CN()
# Path to pretrained model of backbone
_C.TRAIN.TRICK.PRETRAIN_PATH = r'C:\Users\admin\.cache\torch\checkpoints\resnet50-19c8e357.pth'##'modeling/se_resnext50_32x4d-a260b3a4.pth'
# If train with label smooth, options: 'on', 'off'
_C.TRAIN.TRICK.IF_LABELSMOOTH = 'on'

_C.TRAIN.TRICK.PRETRAINED = 1



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specific Setting - Val Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.VAL = CN()
# If test with re-ranking, options: 'yes','no'
_C.VAL.RE_RANKING = 'no'
# Path to trained model
_C.VAL.WEIGHT = ""
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.VAL.FEAT_NORM = 'yes'
# -----------------------------------------------------------------------------
# Val Transform
# -----------------------------------------------------------------------------
#_C.VAL.TRANSFORM = CN()
# Random probability for image horizontal flip
#_C.VAL.TRANSFORM.PROB = 0.5
# Random probability for random erasing
#_C.VAL.TRANSFORM.RE_PROB = 0.5
# -----------------------------------------------------------------------------
# Val Dataloader
# -----------------------------------------------------------------------------
_C.VAL.DATALOADER = CN()
# Number of categories per batch
_C.VAL.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per category in a batch
_C.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of images per batch
_C.VAL.DATALOADER.IMS_PER_BATCH = _C.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * _C.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
# Number of Mask per batch
_C.VAL.DATALOADER.MASK_PER_BATCH = 1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specific Setting - Test Configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.TEST = CN()
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# -----------------------------------------------------------------------------
# Test Transform
# -----------------------------------------------------------------------------
#_C.TEST.TRANSFORM = CN()
# Random probability for image horizontal flip
#_C.TEST.TRANSFORM.PROB = 0.5
# Random probability for random erasing
#_C.TEST.TRANSFORM.RE_PROB = 0.5
# -----------------------------------------------------------------------------
# Test Dataloader
# -----------------------------------------------------------------------------
_C.TEST.DATALOADER = CN()
# Number of categories per batch
_C.TEST.DATALOADER.CATEGORIES_PER_BATCH = 6
# Number of images per category in a batch
_C.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH = 4
# Number of images per batch
_C.TEST.DATALOADER.IMS_PER_BATCH = _C.TEST.DATALOADER.CATEGORIES_PER_BATCH * _C.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
# Number of Mask per batch
_C.TEST.DATALOADER.MASK_PER_BATCH = 1
