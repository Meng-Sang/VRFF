"""
The data configs will be the last one merged into the main configs.
Setups in data configs will override all existed setups!
"""

from yacs.config import CfgNode as CN
_CN = CN()
_CN.DATASET = CN()
_CN.TRAINER = CN()

# training data configs
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
# validation set configs
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None
_CN.DATASET.VAL_INTRINSIC_PATH = None

# testing data configs
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None
_CN.DATASET.TEST_INTRINSIC_PATH = None

# dataset configs
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0  # for both test and val

cfg = _CN
