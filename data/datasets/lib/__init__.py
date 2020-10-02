# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
custom datasets library
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
#from data.datasets.lib.fundusTR import FundusTR
from .ddr import DDR_DR_GRADING, DDR_LESION_SEGMENTATION, DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION, DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION_COLORMASK
from .fundusTRjoint import FundusTR_DRgrading, FundusTR_DRgrading_WeakSupervision
from .examples import Examples

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    #'fundusTR': FundusTR,   # 最初  弃
    'ddr_dr_grading': DDR_DR_GRADING,
    'ddr_lesion_segmentation_regroup': DDR_LESION_SEGMENTATION,
    'ddr_lesion_segmentation_multilabel_weaksupervision': DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION,
    'ddr_lesion_segmentation_multilabel_weaksupervision_colormask': DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION_COLORMASK,
    'examples':  Examples,
    'fundusTR': FundusTR_DRgrading,
    'fundusTRjoint': FundusTR_DRgrading_WeakSupervision,
}

def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)