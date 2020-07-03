# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .dataset_loader import ImageDataset, SegmentationDataset

from data.datasets.lib.cuhk03 import CUHK03
from data.datasets.lib.dukemtmcreid import DukeMTMCreID
from data.datasets.lib.market1501 import Market1501
from data.datasets.lib.msmt17 import MSMT17
from data.datasets.lib.fundusTR import FundusTR
from data.datasets.lib.ddr_DRgrading import DDR_DRgrading, DDR_DRgrading_WeakSupervision
from data.datasets.lib.examples import Examples

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'fundusTR': FundusTR,
    'ddr_DRgrading': DDR_DRgrading,
    'ddr_DRgrading_WeakSupervision':  DDR_DRgrading_WeakSupervision,
    'examples':  Examples,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
