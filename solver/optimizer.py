# encoding: utf-8
"""
@author:  cjy
@contact: sychenjiayang@163.com
"""

import torch
from collections import defaultdict





# 创建多个optimizer，用来交替训练模型的各个子部分
def make_optimizers(cfg, model, bias_free = False):
    params_dict2 = {}
    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        params_dict2[name] = parameters.detach().cpu().numpy()

    #model的参数
    # 1. 按关键字进行分组
    groupKeys = ["classifier", "others"]
    params_dict = defaultdict(list)
    parameters = model.named_parameters()
    for key, value in parameters:
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.SCHEDULER.BASE_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.SCHEDULER.BASE_LR * cfg.SOLVER.SCHEDULER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS

            # bias-free CJY
            if bias_free == True:
                if "classifier.6." not in key:
                    torch.nn.init.constant_(value, 0.0)
                    continue

        for gkey in groupKeys:
            if gkey in key or gkey == "others":
                params_dict[gkey].append({"params": [value], "lr": lr, "weight_decay": weight_decay})
                break

    # 2. 将关键字组再次分组  例：[[0,1], [2]]
    gkeys_divided_list = [[0, 1]]   #将上述group再次分组
    params_divided_list = []
    for sub in gkeys_divided_list:
        p = []
        for i in sub:
            p += params_dict[groupKeys[i]]
        params_divided_list.append(p)

    # 3. group-wise optimizer
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
