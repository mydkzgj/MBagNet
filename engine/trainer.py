# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

import torch
import torch.nn as nn

import torchvision

from ignite.engine import Engine, Events
from ignite.handlers import Timer #ModelCheckpoint,  #自己编写一下ModelCheckpoint，使其可以设置初始轮数
from solver.checkpoint import ModelCheckpoint

from ignite.metrics import RunningAverage
from ignite.metrics import Accuracy
from ignite.metrics import Precision

from engine.evaluator import do_inference

from torch.utils.tensorboard import SummaryWriter

from solver import WarmupMultiStepLR

import torch.nn.functional as F
import copy

"""
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError(
        "No tensorboardX package is found. Please install with the command: \npip install tensorboardX")
"""


#CJY at 2019.9.24 既然这里面定义的与inference.py 中的一样能不能直接引用
#还没加

global ITER
ITER = 0

global epochs_traverse_optimizers
epochs_traverse_optimizers = 0

global epochs_per_optimizer
epochs_per_optimizer = 500

global op2loss
op2loss = {0:"D", 1:"G"}

global weight
weight = 1


global model2
model2 = None

def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys() :
        #if (not k in wf)):
        if ((not k in wf) | (k=='fc.weight') | (k=='fc.bias')):
            wf[k] = wt[k]
    model_to.load_state_dict(wf)


# 创建multilabel
def label2multilabel(label):
    num_classes = 6
    label_groups = [[0, 1, 2], [3, 4, 5]]

    #创建label到对应的multilabel tensor的字典
    label_dict = {}
    for lg in label_groups:
        t = torch.zeros(1, num_classes)
        for l in lg:
            t[0][l] = 1
        for l in lg:
            label_dict[l] = t

    #依据label创建multilabel
    for i in range(label.shape[0]):
        if i == 0:
            multilabel = label_dict[label[i].item()]
        else:
            multilabel = torch.cat([multilabel, label_dict[label[i].item()]], dim=0)

    return multilabel



def create_supervised_trainer(model, optimizers, metrics, loss_fn, device=None,):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        #optimizer加载进来的是cpu类型，需要手动转成gpu。
        for state in optimizers[0].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if model.gradCAMType != "none":
            model2 = copy.deepcopy(model)

    epochs_traverse_optimizers = len(optimizers) * epochs_per_optimizer

    def _update(engine, batch):
        model.train()

        # 分为D和G不同的优化器（暂时无用）
        schedulers_epochs_index = (engine.state.epoch - 1) // epochs_traverse_optimizers
        index = (engine.state.epoch - 1) % epochs_traverse_optimizers  # 注意engine.state.epoch从1开始
        phase_index = index // epochs_per_optimizer
        # 记录在state中，方便传递给handler
        engine.state.optimizer_index = phase_index#(phase_index + 1) % len(optimizers)  #加了一个偏置，先训练G
        engine.state.schedulers_epochs_index = schedulers_epochs_index
        engine.state.epochs_traverse_optimizers = epochs_traverse_optimizers
        engine.state.losstype = op2loss[engine.state.optimizer_index]

        # 获取数据
        grade_imgs, grade_labels, seg_imgs, seg_masks, seg_labels = batch   #这个格式应该跟collate_fn的处理方式对应
        #seg_masks = torch.gt(seg_masks, 0).float()
        # 记录grade和seg的样本数量
        grade_num = grade_imgs.shape[0]
        seg_num = seg_masks.shape[0]
        # 将grade和seg样本concat起来
        imgs = torch.cat([grade_imgs, seg_imgs], dim=0)
        labels = torch.cat([grade_labels, seg_labels], dim=0)
        # 置入cuda
        #one_hot_labels = torch.nn.functional.one_hot(grade_labels, scores.shape[1]).float()
        #grade_imgs = grade_imgs.to(device) if torch.cuda.device_count() >= 1 else grade_imgs
        grade_labels = grade_labels.to(device) if torch.cuda.device_count() >= 1 else grade_labels
        seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 else seg_labels
        seg_masks = seg_masks.to(device) if torch.cuda.device_count() >= 1 else seg_masks
        imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
        labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels

        # 运行模型
        # 设定有多少样本需要进行支路的运算
        model.transimitBatchDistribution((grade_num+seg_num-model.branch_img_num, model.branch_img_num))
        model.transmitClassifierWeight()   #如果是BOF 会回传分类器权重
        #if model.gradCAMType == True and model.target_layer == "":
        #    imgs.requires_grad_(True)
        #model.base.features.denseblock4.eval()

        logits = model(imgs)               #为了减少显存，还是要区分grade和seg
        grade_logits = logits[0:grade_num]

        # 生成Grad-CAM
        if model.gradCAMType != "none":
            # 将label转为one - hot
            one_hot_labels = torch.nn.functional.one_hot(labels, model.num_classes).float()
            one_hot_labels = one_hot_labels.to(device) if torch.cuda.device_count() >= 1 else one_hot_labels
            # 回传one-hot向量
            logits.backward(gradient=one_hot_labels, retain_graph=True)#, create_graph=True)
            # 生成CAM
            gcam_list = []
            target_layer_num = len(model.target_layer)
            maxpool_base_kernel_size = 1 #奇数
            for i in range(target_layer_num):
                inter_output = model.inter_output[i]  # 此处分离节点，别人皆不分离  .detach()
                inter_gradient = model.inter_gradient[target_layer_num - i - 1]
                if i == target_layer_num-1:   #最后一层是denseblock4的输出
                    gcam = F.conv2d(inter_output, model.classifier.weight.unsqueeze(-1).unsqueeze(-1))
                    pick_label = labels[grade_num + seg_num - model.branch_img_num:grade_num + seg_num]
                    pick_list = []
                    for j in range(pick_label.shape[0]):
                        pick_list.append(gcam[j, pick_label[j]].unsqueeze(0).unsqueeze(0))
                    gcam = torch.cat(pick_list, dim=0)

                    # 为了降低与掩膜对齐的强硬度，特地增加了Maxpool操作
                    #maxpool_kernel_size = maxpool_base_kernel_size + pow(2, (target_layer_num - i))
                    #gcam = F.max_pool2d(gcam, kernel_size=maxpool_kernel_size, stride=1, padding=maxpool_kernel_size // 2)
                    #gcam = torch.sigmoid(gcam)
                    pos = torch.gt(gcam, 0).float()
                    gcam_pos = gcam * pos
                    gcam_neg = gcam * (1 - pos)

                    gcam_pos_abs_max = torch.max(gcam_pos.view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(
                        -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
                    gcam_neg_abs_max = torch.max(gcam_neg.abs().view(gcam.shape[0], -1), dim=1)[0].clamp(
                        1E-12).unsqueeze(
                        -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)

                    gcam_pos_mean = (torch.sum(gcam_pos) / torch.sum(pos).clamp(min=1E-12)) * 0.9

                    sigma = 0.5
                    gcam = (1 - torch.relu(-gcam_pos / (gcam_pos_abs_max.clamp(
                        min=1E-12).detach() * sigma) + 1)) + gcam_neg / gcam_neg_abs_max.clamp(min=1E-12).detach()
                    # gcam = torch.tanh(gcam_pos/gcam_pos_mean.clamp(min=1E-12).detach()) + gcam_neg/gcam_neg_abs_max.clamp(min=1E-12).detach()
                    gcam = gcam / 2 + 0.5

                else:
                    #avg_gradient = torch.nn.functional.adaptive_avg_pool2d(model.inter_gradient, 1)
                    gcam = torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
                    # 为了降低与掩膜对齐的强硬度，特地增加了Maxpool操作
                    #maxpool_kernel_size = maxpool_base_kernel_size + pow(2, (target_layer_num - i))
                    #gcam = F.max_pool2d(gcam, kernel_size=maxpool_kernel_size, stride=1, padding=maxpool_kernel_size//2)
                    #标准化
                    """
                    gcam_flatten = gcam.view(gcam.shape[0], -1)
                    gcam_var = torch.var(gcam_flatten, dim=1).detach()
                    gcam = gcam/gcam_var
                    gcam = torch.sigmoid(gcam)
                    #"""
                    #"""
                    pos = torch.gt(gcam, 0).float()
                    gcam_pos = gcam * pos
                    gcam_neg = gcam * (1-pos)

                    gcam_pos_abs_max = torch.max(gcam_pos.view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(
                        -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
                    gcam_neg_abs_max = torch.max(gcam_neg.abs().view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(
                        -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)

                    gcam_pos_mean = (torch.sum(gcam_pos) / torch.sum(pos).clamp(min=1E-12)) * 0.9

                    sigma = 0.5
                    gcam = (1-torch.relu(-gcam_pos/(gcam_pos_abs_max.clamp(min=1E-12).detach() * sigma)+1)) + gcam_neg / gcam_neg_abs_max.clamp(min=1E-12).detach()
                    #gcam = torch.tanh(gcam_pos/gcam_pos_mean.clamp(min=1E-12).detach()) + gcam_neg/gcam_neg_abs_max.clamp(min=1E-12).detach()
                    gcam = gcam/2 + 0.5

                    #gcam_max = torch.max(torch.relu(gcam).view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
                    #gcam_min = torch.min(gcam.view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
                    #gcam = torch.relu(gcam) / gcam_max.detach()
                    #"""
                    """
                    gcam_flatten = torch.relu(gcam).view(gcam.shape[0], -1)  # 负的也算上吧
                    gcam_gt0 = torch.gt(gcam_flatten, 0).float()
                    gcam_sum = torch.sum(gcam_flatten, dim=-1)
                    gcam_sum_num = torch.sum(gcam_gt0, dim=-1)
                    gcam_mean = gcam_sum / gcam_sum_num.clamp(min=1E-12) * 0.9

                    gcam = gcam / gcam_mean.clamp(min=1E-12).detach()
                    gcam = torch.sigmoid(gcam)
                    #"""

                # 插值
                gcam = torch.nn.functional.interpolate(gcam, (seg_masks.shape[-2], seg_masks.shape[-1]), mode='bilinear')  #mode='nearest'  'bilinear'
                gcam_list.append(gcam)   #将不同模块的gcam保存到gcam_list中

            overall_gcam = torch.cat(gcam_list, dim=1)
            #overall_gcam = torch.max(overall_gcam, dim=1, keepdim=True)[0]
            overall_gcam = torch.mean(overall_gcam, dim=1)
            gcam_list = [overall_gcam]



            # 将这些gcam 扩增 并且 fusion

            # GAIN论文中 生成soft_mask的做法
            #sigma = 1/target_layer_num#0.5
            #w = 8
            #gcam = torch.sigmoid(w * (gcam - sigma))

            for op in optimizers:
                op.zero_grad()

            model.inter_output.clear()
            model.inter_gradient.clear()


        #CJY at 2020.3.5 masked img reload
        # 掩膜图像重新输入
        if model.maskedImgReloadType != "none":
            #1.生成soft_mask
            if model.maskedImgReloadType == "seg_mask":
                if model.segmentationType != "denseFC":
                    raise Exception("segmentationType can't match maskedImgReloadType")
                if model.seg_num_classes != 1:
                    soft_mask = torch.max(model.base.seg_attention, dim=1, keepdim=True)[0]
                else:
                    soft_mask = model.base.seg_attention
                soft_mask = torch.sigmoid(soft_mask)
                #soft_mask = torch.nn.functional.max_pool2d(soft_mask, kernel_size=31, stride=1, padding=15)
            elif model.maskedImgReloadType == "gradcam_mask":   #生成grad-cam
                #if model.gradCAMType != "reload":
                #    raise Exception("segmentationType can't match maskedImgReloadType")
                soft_mask = overall_gcam
            elif model.maskedImgReloadType == "joint_mask":   #生成grad-cam:
                if model.segmentationType != "denseFC":
                    raise Exception("segmentationType can't match maskedImgReloadType")
                #soft_mask = torch.cat([torch.sigmoid(model.base.seg_attention), gcam], dim=1)
                soft_mask = torch.cat([seg_masks, gcam], dim=1)   # 将分割结果替换成真正标签
                soft_mask = torch.max(soft_mask, dim=1, keepdim=True)[0].detach()

                #soft_mask = torch.nn.functional.max_pool2d(soft_mask, kernel_size=501, stride=1, padding=250)
                #soft_mask = torch.nn.functional.avg_pool2d(soft_mask, kernel_size=81, stride=1, padding=40)
            else:
                pass
            # 2.生成masked_img
            rimgs = imgs[model.batchDistribution[0]:model.batchDistribution[0] + model.batchDistribution[1]]

            #pos_masked_img = soft_mask * rimgs
            neg_masked_img = (1-soft_mask) * rimgs
            # 3.reload maskedImg
            transfer_weights(model, model2)
            model2.eval()
            model2.transimitBatchDistribution(0)
            pm_logits = None#model(pos_masked_img)
            nm_logits = model2(neg_masked_img)
        else:
            pm_logits = None
            nm_logits = None

        # 确定分割结果输出类型
        if model.segmentationType == "denseFC":
            output_masks = model.base.seg_attention#[model.base.seg_attention.shape[0]-seg_num: model.base.seg_attention.shape[0]]
            if model.segSupervisedType == "strong":
                seg_masks = seg_masks
                gcam_masks = None
            elif model.segSupervisedType == "weak":
                gcam_masks = gcam#[gcam.shape[0] - seg_num:gcam.shape[0]]
                seg_masks = None
            elif model.segSupervisedType == "joint":
                gcam_masks = gcam_list#[gcam.shape[0] - seg_num:gcam.shape[0]]
                seg_masks = seg_masks#torch.cat([seg_masks, gcam_masks], dim=1)
            elif model.segSupervisedType == "none":
                gcam_masks = None
                seg_masks = None
        elif model.segmentationType == "gradCAM":
            output_masks = gcam[gcam.shape[0]-seg_num: gcam.shape[0]]
        else:
            output_masks = None

        # for show loss 计算想查看的loss
        #forShow = torch.mean(torch.sigmoid(torch.max(model.base.seg_attention, dim=1, keepdim=True)[0]))
        forShow = torch.mean(overall_gcam)
        #forShow = torch.mean(soft_mask)

        # 计算loss
        #利用不同的optimizer对模型中的各子模块进行分阶段优化。目前最简单的方式是周期循环启用optimizer
        losses = loss_fn[engine.state.losstype](logit=logits, label=labels, output_mask=output_masks, seg_mask=seg_masks, seg_label=seg_labels, gcam_mask=gcam_masks, pos_masked_logit=pm_logits, neg_masked_logit=nm_logits, show=forShow)    #损失词典
        #为了减少"pos_masked_img_loss" 和 "cross_entropy_loss"之间的冲突，特设定动态weight，使用 "cross_entropy_loss" detach
        #pos_masked_img_loss_weight = 1/(1+losses["cross_entropy_loss"].detach())
        weight = {"cross_entropy_loss":1, "seg_mask_loss":0.2, "gcam_mask_loss":0.6, "pos_masked_img_loss":1, "neg_masked_img_loss":0, "for_show_loss":0}
        gl_weight = [1, 0.8, 0.6, 0.4]
        loss = 0
        for lossKey in losses.keys():
            if lossKey == "gcam_mask_loss":
                gcam_loss = 0
                for index, gl in enumerate(losses[lossKey]):
                    gcam_loss = gcam_loss + gl * gl_weight[index]
                loss = loss + gcam_loss * weight[lossKey]
            else:
                loss += losses[lossKey] * weight[lossKey]
        loss = loss/model.accumulation_steps

        # 反向传播
        loss.backward()
        # 参数优化
        if engine.state.iteration % model.accumulation_steps == 0:  # 此处要注意
            optimizers[engine.state.optimizer_index].step()
            for op in optimizers:
                op.zero_grad()

        # compute acc
        #acc = (scores.max(1)[1] == grade_labels).float().mean()
        return {"scores": grade_logits, "labels": grade_labels, "losses": losses, "total_loss": loss.item()}
    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        classes_list,
        optimizers,
        schedulers,
        loss_fn,
        start_epoch,
):
    #1.先把cfg中的参数导出
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.SOLVER.OUTPUT_DIR
    device = cfg.MODEL.DEVICE

    #2.构建模块
    logger = logging.getLogger("fundus_prediction.train")
    logger.info("Start training")

    # TensorBoard setup
    writer_train = {}
    for i in range(len(optimizers)):
        writer_train[i] = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/train/" + str(i))

    writer_val = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/val")

    writer_train["graph"] = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/train/graph")

    try:
        #print(model)
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        writer_train["graph"].add_image('images', grid, 0)
        writer_train["graph"].add_graph(model, images)
        writer_train["graph"].flush()
    except Exception as e:
        print("Failed to save model graph: {}".format(e))



    # 设置训练相关的metrics
    metrics_train = {"avg_total_loss": RunningAverage(output_transform=lambda x: x["total_loss"]),
                     "avg_precision": RunningAverage(Precision(output_transform=lambda x: (x["scores"], x["labels"]))),
                     "avg_accuracy": RunningAverage(Accuracy(output_transform=lambda x: (x["scores"], x["labels"]))),  #由于训练集样本均衡后远离原始样本集，故只采用平均metric
                     }

    lossKeys = cfg.LOSS.TYPE.split(" ")
    # 设置loss相关的metrics
    for lossName in lossKeys:
        if lossName == "similarity_loss":
            metrics_train["AVG-" + "similarity_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["similarity_loss"])
        elif lossName == "ranked_loss":
            metrics_train["AVG-" + "ranked_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["ranked_loss"])
        elif lossName == "cranked_loss":
            metrics_train["AVG-" + "cranked_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["cranked_loss"])
        elif lossName == "cross_entropy_loss":
            metrics_train["AVG-" + "cross_entropy_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["cross_entropy_loss"])
        elif lossName == "cluster_loss":
            metrics_train["AVG-" + "cluster_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["cluster_loss"][0])
        elif lossName == "one_vs_rest_loss":
            metrics_train["AVG-" + "one_vs_rest_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["one_vs_rest_loss"])
        elif lossName == "attention_loss":
            metrics_train["AVG-" + "attention_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["attention_loss"])
        elif lossName == "class_predict_loss":
            metrics_train["AVG-" + "class_predict_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["class_predict_loss"])
        elif lossName == "kld_loss":
            metrics_train["AVG-" + "kld_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["kld_loss"])
        elif lossName == "margin_loss":
            metrics_train["AVG-" + "margin_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["margin_loss"])
        elif lossName == "cross_entropy_multilabel_loss":
            metrics_train["AVG-" + "cross_entropy_multilabel_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["cross_entropy_multilabel_loss"])
        elif lossName == "seg_mask_loss":
            metrics_train["AVG-" + "seg_mask_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["seg_mask_loss"])
        elif lossName == "gcam_mask_loss":
            #metrics_train["AVG-" + "gcam_mask_loss"] = RunningAverage(
            #    output_transform=lambda x: x["losses"]["gcam_mask_loss"])
            metrics_train["AVG-" + "gcam_mask_loss0"] = RunningAverage(
                output_transform=lambda x: x["losses"]["gcam_mask_loss"][0])
            metrics_train["AVG-" + "gcam_mask_loss1"] = RunningAverage(
                output_transform=lambda x: x["losses"]["gcam_mask_loss"][1])
            metrics_train["AVG-" + "gcam_mask_loss2"] = RunningAverage(
                output_transform=lambda x: x["losses"]["gcam_mask_loss"][2])
            metrics_train["AVG-" + "gcam_mask_loss3"] = RunningAverage(
                output_transform=lambda x: x["losses"]["gcam_mask_loss"][3])
        elif lossName == "pos_masked_img_loss":
            metrics_train["AVG-" + "pos_masked_img_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["pos_masked_img_loss"])
        elif lossName == "neg_masked_img_loss":
            metrics_train["AVG-" + "neg_masked_img_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["neg_masked_img_loss"])
        elif lossName == "for_show_loss":
            metrics_train["AVG-" + "for_show_loss"] = RunningAverage(
                output_transform=lambda x: x["losses"]["for_show_loss"])
        else:
            raise Exception('expected METRIC_LOSS_TYPE should be similarity_loss, ranked_loss, cranked_loss'
                            'but got {}'.format(cfg.LOSS.TYPE))


    trainer = create_supervised_trainer(model, optimizers, metrics_train, loss_fn, device=device,)

    #CJY  at 2019.9.26
    def output_transform(output):
        # `output` variable is returned by above `process_function`
        y_pred = output['scores']
        y = output['labels']
        return y_pred, y  # output format is according to `Accuracy` docs

    metrics_eval = {"overall_accuracy": Accuracy(output_transform=output_transform),
                    "precision": Precision(output_transform=output_transform)}

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=300, require_empty=False, start_step=start_epoch)
    #checkpointer_save_graph = ModelCheckpoint(output_dir, cfg.MODEL.NAME+"_graph", checkpoint_period, n_saved=300, require_empty=False, start_step=start_epoch, save_as_state_dict=False)
    timer = Timer(average=True)

    #3.将模块与engine联系起来attach
    #CJY at 2019.9.23
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizers[0]})

    #trainer.add_event_handler(Events.STARTED, checkpointer, {'model': model,
    #                                                                 'optimizer': optimizers[0]})
    #trainer.add_event_handler(Events.STARTED, checkpointer_save_graph, {'model': model,
    #                                                                 'optimizer': optimizers[0]})
    #torch.save(model, output_dir + "/" + cfg.MODEL.NAME+"_graph.pkl")

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)


    #4.事件处理函数
    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        engine.state.iteration = engine.state.iteration + start_epoch * len(train_loader)

        logger.info("Model:{}".format(model))
        logger.info("Model:{}".format(model.count_param()))
        inputshape = (3, cfg.DATA.TRANSFORM.SIZE[0], cfg.DATA.TRANSFORM.SIZE[1])
        logger.info("Model:{}".format(model.count_param2(input_shape=inputshape)))
        #print(model)
        #print(model.count_param())
        #print(model.count_param2())

        """
        metrics = do_inference(cfg, model, val_loader, classes_list, loss_fn, plotFlag=False)

        step = 0#len(train_loader) * (engine.state.epoch - 1) + engine.state.iteration
        for preKey in metrics['precision'].keys():
            writer_val.add_scalar("Precision/" + str(preKey), metrics['precision'][preKey], step)

        for recKey in metrics['recall'].keys():
            writer_val.add_scalar("Recall/" + str(recKey), metrics['recall'][recKey], step)

        for aucKey in metrics['roc_auc'].keys():
            writer_val.add_scalar("ROC_AUC/" + str(aucKey), metrics['roc_auc'][aucKey], step)

        writer_val.add_scalar("OverallAccuracy", metrics["overall_accuracy"], step)

        # writer.add_scalar("Val/"+"confusion_matrix", metrics['confusion_matrix'], step)

        # 混淆矩阵 和 ROC曲线可以用图的方式来存储
        roc_numpy = metrics["roc_figure"]
        writer_val.add_image("ROC", roc_numpy, step, dataformats='HWC')

        confusion_matrix_numpy = metrics["confusion_matrix_numpy"]
        writer_val.add_image("ConfusionMatrix", confusion_matrix_numpy, step, dataformats='HWC')

        writer_val.flush()
        #"""

    @trainer.on(Events.EPOCH_COMPLETED) #_STARTED)   #注意，在pytorch1.2里面 scheduler.steo()应该放到 optimizer.step()之后
    def adjust_learning_rate(engine):
        """
        #if (engine.state.epoch - 1) % engine.state.epochs_traverse_optimizers == 0:
        if engine.state.epoch == 2:
            op_i_scheduler1 = WarmupMultiStepLR(optimizers[0], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                               cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                               cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD)
            op_i_scheduler2 = WarmupMultiStepLR(optimizers[1], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                                cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                                cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD)
            engine.state.schedulers = [op_i_scheduler1, op_i_scheduler2]
            print("copy")
        """
        schedulers[engine.state.schedulers_epochs_index][engine.state.optimizer_index].step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % (log_period*model.accumulation_steps) == 0:
            step = engine.state.iteration

            #写入train-summary
            #记录avg-presicion
            avg_precision = engine.state.metrics['avg_precision'].numpy().tolist()
            avg_precisions = {}
            ap_sum = 0
            for index, ap in enumerate(avg_precision):
                avg_precisions[index] = float("{:.2f}".format(ap))
                ap_sum += avg_precisions[index]
                scalarDict = {}
                for i in range(len(optimizers)):
                    if i != engine.state.optimizer_index:
                        scalarDict["optimizer" + str(i)] = 0
                    else:
                        scalarDict["optimizer" + str(i)] = avg_precisions[index]
                    writer_train[i].add_scalar("Precision/" + str(index), scalarDict["optimizer" + str(i)], step)
                    writer_train[i].flush()
            avg_precisions["avg_precision"] = float("{:.2f}".format(ap_sum/len(avg_precision)))

            #记录avg-loss
            avg_losses = {}
            for lossName in lossKeys:
                if lossName == "gcam_mask_loss":
                    for i in range(4):
                        avg_losses[lossName+str(i)] = (float("{:.3f}".format(engine.state.metrics["AVG-" + lossName + str(i)])))
                        scalarDict = {}
                        for j in range(len(optimizers)):
                            if j != engine.state.optimizer_index:
                                scalarDict["optimizer" + str(j)] = 0
                            else:
                                scalarDict["optimizer" + str(j)] = avg_losses[lossName+str(i)]
                            writer_train[j].add_scalar("Loss/" + lossName + str(i), scalarDict["optimizer" + str(j)], step)
                            writer_train[j].flush()
                else:
                    avg_losses[lossName] = (float("{:.3f}".format(engine.state.metrics["AVG-" + lossName])))
                    scalarDict = {}
                    for i in range(len(optimizers)):
                        if i != engine.state.optimizer_index:
                            scalarDict["optimizer" + str(i)] = 0
                        else:
                            scalarDict["optimizer" + str(i)] = avg_losses[lossName]
                        writer_train[i].add_scalar("Loss/" + lossName, scalarDict["optimizer" + str(i)], step)
                        writer_train[i].flush()


            #记录其余标量
            scalar_list = ["avg_accuracy", "avg_total_loss"]
            for scalar in scalar_list:
                scalarDict = {}
                for i in range(len(optimizers)):
                    if i != engine.state.optimizer_index:
                        scalarDict["optimizer" + str(i)] = 0
                    else:
                        scalarDict["optimizer" + str(i)] = engine.state.metrics[scalar]
                    writer_train[i].add_scalar("Train/" + scalar, scalarDict["optimizer" + str(i)], step)
                    writer_train[i].flush()

            #记录学习率
            LearningRateDict = {}
            for i in range(len(optimizers)):
                if i != engine.state.optimizer_index:
                    LearningRateDict["optimizer" + str(i)] = 0
                else:
                    LearningRateDict["optimizer" + str(i)] = schedulers[engine.state.schedulers_epochs_index][engine.state.optimizer_index].get_lr()[0]
                writer_train[i].add_scalar("Train/" + "LearningRate", LearningRateDict["optimizer" + str(i)], step)
                writer_train[i].flush()

            #记录weight
            choose_list = ["base.conv1.weight", "base.bn1.weight",
                          "base.layer1.0.conv1.weight", "base.layer1.2.conv3.weight",
                          "base.layer2.0.conv1.weight", "base.layer2.3.conv3.weight",
                          "base.layer3.0.conv1.weight", "base.layer3.5.conv3.weight",
                          "base.layer4.0.conv1.weight", "base.layer4.2.conv1.weight",
                          "bottleneck.weight", "classifier.weight"]
            """
            #记录参数分布 非常耗时
            params_dict = {}
            for name, parameters in model.named_parameters():
                #print(name, ':', parameters.size())
                params_dict[name] = parameters.detach().cpu().numpy()
            #print(len(params_dict))
                        
            for cp in params_dict.keys():
                writer_train["graph"].add_histogram("Train/" + cp, params_dict[cp], step)
                writer_train["graph"].flush()
            #"""

            logger.info("Epoch[{}] Iteration[{}/{}] Training {} - ATLoss: {:.3f}, AvgLoss: {}, Avg Pre: {}, Avg_Acc: {:.3f}, Base Lr: {:.2e}, step: {}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.losstype,
                                engine.state.metrics['avg_total_loss'], avg_losses, avg_precisions, engine.state.metrics['avg_accuracy'],
                                schedulers[engine.state.schedulers_epochs_index][engine.state.optimizer_index].get_lr()[0], step))

            #logger.info(engine.state.output["rf_loss"])

            if engine.state.output["losses"].get("cluster_loss") != None:
                logger.info("Epoch[{}] Iteration[{}/{}] Center {} \n r_inter: {}, r_outer: {}, step: {}"
                            .format(engine.state.epoch, ITER, len(train_loader),
                                    engine.state.output["losses"]["cluster_loss"][-1]["center"].cpu().detach().numpy(),
                                    engine.state.output["losses"]["cluster_loss"][-1]["r_inter"].item(),
                                    engine.state.output["losses"]["cluster_loss"][-1]["r_outer"].item(),
                                    step))

        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            metrics = do_inference(cfg, model, val_loader, classes_list, loss_fn)

            step = engine.state.iteration
            for preKey in metrics['precision'].keys():
                writer_val.add_scalar("Precision/" + str(preKey), metrics['precision'][preKey], step)

            for recKey in metrics['recall'].keys():
                writer_val.add_scalar("Recall/" + str(recKey), metrics['recall'][recKey], step)

            for aucKey in metrics['roc_auc'].keys():
                writer_val.add_scalar("ROC_AUC/" + str(aucKey), metrics['roc_auc'][aucKey], step)

            writer_val.add_scalar("OverallAccuracy", metrics["overall_accuracy"], step)

            #writer.add_scalar("Val/"+"confusion_matrix", metrics['confusion_matrix'], step)

            #混淆矩阵 和 ROC曲线可以用图的方式来存储
            #roc_numpy = metrics["roc_figure"]
            #writer_val.add_image("ROC", roc_numpy, step, dataformats='HWC')

            #confusion_matrix_numpy = metrics["confusion_matrix_numpy"]
            #writer_val.add_image("ConfusionMatrix", confusion_matrix_numpy, step, dataformats='HWC')

            writer_val.flush()


    #5.engine运行
    trainer.run(train_loader, max_epochs=epochs)
    for key in writer_train.keys():
        writer_train[key].close()
    writer_val.close()
