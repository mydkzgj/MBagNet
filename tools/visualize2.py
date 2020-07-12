# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg

from data import make_data_loader, make_seg_data_loader, WeakSupervisionDataloader
from modeling import build_model
from loss import make_D_loss

from engine.visualizer2 import do_visualization

from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter

from utils.featrueVisualization import showWeight

import random
import numpy as np


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("fundus_prediction", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    #build
    #train_loader, val_loader, test_loader, classes_list = make_data_loader(cfg)
    train_grad_loader, val_grad_loader, test_grad_loader, classes_list = make_data_loader(cfg)
    train_seg_loader, val_seg_loader, test_seg_loader = make_seg_data_loader(cfg)
    train_loader = WeakSupervisionDataloader(train_grad_loader, train_seg_loader, recycling=False)  #True
    val_loader = WeakSupervisionDataloader(val_grad_loader, val_seg_loader, recycling=False)  #Fasle 不循环使用seg_data
    test_loader = WeakSupervisionDataloader(test_grad_loader, test_seg_loader, recycling=False)
    num_classes = len(classes_list)

    model = build_model(cfg)
    model.load_param("Overall", cfg.TEST.WEIGHT)  #Overall

    #CJY 可视化  densenet层与层之间的联系
    #showWeight(model)

    #loss_fn = make_loss(cfg, num_classes)  # modified by gu
    # loss function
    #loss_func = make_loss(cfg, num_classes)  # modified by gu
    #g_loss_func = make_G_loss(cfg, num_classes)
    d_loss_func, lossClasses = make_D_loss(cfg, num_classes)
    loss_funcs = {}
    loss_funcs["G"] = d_loss_func
    loss_funcs["D"] = d_loss_func
    print('Test with the loss type is', cfg.LOSS.TYPE)


    writer_test = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/test")

    model_save_epoch = cfg.TEST.WEIGHT.split('/')[-1].split('.')[0].split('_')[-1]

    if model_save_epoch.isdigit() == True:
        step = len(train_loader) * int(model_save_epoch)
    else:
        step = 0

    metrics = do_visualization(cfg, model, test_loader, classes_list, loss_funcs, plotFlag=True)

    for preKey in metrics['precision'].keys():
        writer_test.add_scalar("Precision/" + str(preKey), metrics['precision'][preKey], step)

    for recKey in metrics['recall'].keys():
        writer_test.add_scalar("Recall/" + str(recKey), metrics['recall'][recKey], step)

    #for aucKey in metrics['roc_auc'].keys():
    #    writer_test.add_scalar("ROC_AUC/" + str(aucKey), metrics['roc_auc'][aucKey], step)

    writer_test.add_scalar("OverallAccuracy", metrics["overall_accuracy"], step)

    # writer.add_scalar("Val/"+"confusion_matrix", metrics['confusion_matrix'], step)

    # 混淆矩阵 和 ROC曲线可以用图的方式来存储
    #roc_numpy = metrics["roc_figure"]
    #writer_test.add_image("ROC", roc_numpy, step, dataformats='HWC')

    #confusion_matrix_numpy = metrics["confusion_matrix_numpy"]
    #writer_test.add_image("ConfusionMatrix", confusion_matrix_numpy, step, dataformats='HWC')

    writer_test.close()


if __name__ == '__main__':
    seed_torch(2018)
    main()
