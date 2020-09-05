# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
import numpy as np
import random

from torch.backends import cudnn
import torchvision
from torchvision import transforms
#from data.transforms import build_transforms

sys.path.append('.')
from config import cfg

from data import make_data_loader, make_seg_data_loader, WeakSupervisionDataloader
from engine.trainer import do_train
from modeling import build_model
from loss import make_G_loss
from loss import make_D_loss

from solver import WarmupMultiStepLR, make_optimizers


from utils.logger import setup_logger

def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(cfg):
    # prepare dataset
    #train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    #CJY at 2019.9.26  利用重新编写的函数处理同仁数据
    train_grad_loader, val_grad_loader, test_grad_loader, cla_classes_list = make_data_loader(cfg, for_train=True)
    train_seg_loader, val_seg_loader, test_seg_loader, seg_classes_list = make_seg_data_loader(cfg, for_train=True) #True
    train_loader = WeakSupervisionDataloader(train_grad_loader, train_seg_loader)
    val_loader = WeakSupervisionDataloader(val_grad_loader, val_seg_loader)
    test_loader = WeakSupervisionDataloader(test_grad_loader, test_seg_loader)
    classes_list = cla_classes_list if cla_classes_list != [] else seg_classes_list
    num_classes = len(classes_list)

    # build model and load parameter
    model = build_model(cfg)
    if cfg.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD == True:
        if cfg.TRAIN.TRICK.PRETRAINED == True:
            model.load_param("Base", cfg.TRAIN.TRICK.PRETRAIN_PATH)
    else:
        if cfg.TRAIN.TRICK.PRETRAINED == True:
            model.load_param("Overall", cfg.TRAIN.TRICK.PRETRAIN_PATH)
    #print(model)

    # loss function
    #loss_func = make_loss(cfg, num_classes)  # modified by gu
    #g_loss_func = make_G_loss(cfg, num_classes)
    d_loss_func, lossClasses = make_D_loss(cfg, num_classes)
    loss_funcs = {}
    loss_funcs["G"] = d_loss_func
    loss_funcs["D"] = d_loss_func
    print('Train with the loss type is', cfg.LOSS.TYPE)

    # build optimizer
    optimizers = make_optimizers(cfg, model, bias_free=False)  #loss里也可能有参数

    print('Train with the optimizer type is', cfg.SOLVER.OPTIMIZER.NAME)

    # build scheduler （断点续传功能暂时有问题）
    if cfg.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD == True:
        start_epoch = 0
        op_epochs = 10   #如果把所有optimizer看作一轮的话，总的轮数
        schedulers = []
        for epoch_index in range(op_epochs):
            op_schedulers = []
            for i in range(len(optimizers)):
                op_i_scheduler = WarmupMultiStepLR(optimizers[i], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                      cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                      cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD)
                op_schedulers.append(op_i_scheduler)
            schedulers.append(op_schedulers)

    else:
        start_epoch = eval(cfg.TRAIN.TRICK.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        op_epochs = 10   #如果把所有optimizer看作一轮的话，总的轮数
        schedulers = []
        for epoch_index in range(op_epochs):
            op_schedulers = []
            for i in range(len(optimizers)):
                op_i_scheduler = WarmupMultiStepLR(optimizers[i], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                              cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                              cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD,)
                                              #start_epoch)   # KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"
                for i in range(start_epoch):
                    op_i_scheduler.step()
                op_schedulers.append(op_i_scheduler)
            schedulers.append(op_schedulers)


    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        classes_list,
        optimizers,
        schedulers,      # modify for using self trained model
        loss_funcs,
        start_epoch,     # add for using self trained model
    )


def main():
    #解析命令行参数,详见argparse模块
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)    #nargs=argparse.REMAINDER是指所有剩余的参数均转化为一个列表赋值给此项

    args = parser.parse_args()
     
    #os.environ()是python用来获取系统相关信息的。如environ[‘HOME’]就代表了当前这个用户的主目录
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    #此处是指如果有类似yaml重新赋值参数的文件在的话会把它读进来。这也是rbgirshick/yacs模块的优势所在——参数与代码分离
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()   #最终要freeze一下，prevent further modification，也就是参数设置在这一步就完成了，后面都不能再改变了

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #logger主要用于输出运行日志，相比print有一定优势。
    logger = setup_logger("fundus_prediction", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    #哦，此处把config文件又专门读了一遍，并输出了出来
    '''
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    '''
    logger.info("Running with config:\n{}".format(cfg))


    #？上面的GPU与CUDA是什么关系，这个参数的意义是？
    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    seed_torch(2018)
    main()
