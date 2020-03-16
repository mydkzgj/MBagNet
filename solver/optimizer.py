# encoding: utf-8
"""
@author:  cjy
@contact: sychenjiayang@163.com
"""

import torch
from collections import defaultdict

# 创建多个optimizer，用来交替训练模型的各个子部分
def make_optimizers(cfg, model):
    params_dict2 = {}
    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        params_dict2[name] = parameters.detach().cpu().numpy()

    #model的参数
    groupKeys = ["denseblock4", "classifier","others"]
    params_dict = defaultdict(list)
    parameters = model.named_parameters()
    out_include = []
    for key, value in parameters:
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.SCHEDULER.BASE_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.SCHEDULER.BASE_LR * cfg.SOLVER.SCHEDULER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS
        for gkey in groupKeys:
            if gkey in key or gkey == "others":
                if gkey == "base.features1.nonlocal2D.cluster_out":
                    params_dict[gkey].append({"params": [value], "lr": lr, "weight_decay": weight_decay})
                    break
                params_dict[gkey].append({"params": [value], "lr": lr, "weight_decay": weight_decay})
                break

    gkeys_divided_list = [[2]]#,[0,1,2,3]] #[0,1,2,3,4,5,6,7],  3,4,5,6,7,8
    params_divided_list = []
    for sub in gkeys_divided_list:
        p = []
        for i in sub:
            p += params_dict[groupKeys[i]]
        params_divided_list.append(p)

    optimizers = []
    if cfg.SOLVER.OPTIMIZER.NAME == 'SGD':
        for pd in params_divided_list:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER.NAME)(pd , momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)
            optimizers.append(optimizer)
    else:
        for pd in params_divided_list:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER.NAME)(pd)
            optimizers.append(optimizer)

    return optimizers

# 创建多个optimizer，用来交替训练模型的各个子部分
def make_optimizers_for_loss(cfg, loss):
    """
    params_dict2 = {}
    for name, parameters in loss.named_parameters():
        # print(name, ':', parameters.size())
        params_dict2[name] = parameters.detach().cpu().numpy()

    #model的参数
    groupKeys = ["others"]
    params_dict = defaultdict(list)
    parameters = loss.named_parameters()
    out_include = []
    for key, value in parameters:
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.SCHEDULER.LOSS_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        for gkey in groupKeys:
            if gkey in key or gkey == "others":
                params_dict[gkey].append({"params": [value], "lr": lr, "weight_decay": weight_decay})
                break

    gkeys_divided_list = [[0]]#,[0,1,2,3]] #[0,1,2,3,4,5,6,7],  3,4,5,6,7,8
    params_divided_list = []
    for sub in gkeys_divided_list:
        p = []
        for i in sub:
            p += params_dict[groupKeys[i]]
        params_divided_list.append(p)
    """
    for key in loss.keys():
        lr = cfg.SOLVER.SCHEDULER.LOSS_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        if key == "cluster_loss":
            params_dict = defaultdict(list)
            if loss[key].centers.requires_grad:
                params_dict[key].append({"params": [loss[key].centers], "lr": lr, "weight_decay": weight_decay})
            #if loss[key].R.requires_grad:
                #params_dict[key].append({"params": [loss[key].R], "lr": lr, "weight_decay": weight_decay})
            #if loss[key].R_P.requires_grad:
            #    params_dict[key].append({"params": [loss[key].R_P], "lr": lr, "weight_decay": weight_decay})
            if loss[key].r_P.requires_grad:
                params_dict[key].append({"params": [loss[key].r_P], "lr": lr, "weight_decay": weight_decay})

    loss_optimizer = torch.optim.SGD(params_dict["cluster_loss"])
    return loss_optimizer