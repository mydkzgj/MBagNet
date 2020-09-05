# encoding: utf-8

import torch.nn.functional as F
import torch
from .reanked_loss import RankedLoss
from .reanked_clu_loss import CRankedLoss
from .common_loss import CommonLoss
from .similarity_loss import SimilarityLoss
from .class_predict_loss import ClassPredictLoss
from .cluster_loss import ClusterLoss
from .one_vs_rest_loss import OneVsRestLoss
from .attention_loss import AttentionLoss
from torch.nn import CrossEntropyLoss
from torch.nn import NLLLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import KLDivLoss
from .margin_loss import MarginLoss
from .cross_entropy_label_smooth import CrossEntropyLabelSmooth
from .multilabel_binary_cross_entropy import MultilabelBinaryCrossEntropy
from .mask_loss import SegMaskLoss, GradCamMaskLoss
from .masked_img_loss import PosMaskedImgLoss, NegMaskedImgLoss
from .forshow_loss import ForShowLoss
from .mse_loss import MSELoss




def make_D_loss(cfg, num_classes):

    sampler = cfg.DATA.DATALOADER.SAMPLER

    lossKeys = cfg.LOSS.TYPE.split(" ")

    #创建loss的类
    lossClasses = {}
    for lossName in lossKeys:
        if lossName == "similarity_loss":
            similarity_loss = SimilarityLoss()
            lossClasses["similarity_loss"] = similarity_loss
        elif lossName == "ranked_loss":
            ranked_loss = RankedLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # ranked_loss
            lossClasses["ranked_loss"] = ranked_loss
        elif lossName == "cranked_loss":
            cranked_loss = CRankedLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # cranked_loss
            lossClasses["cranked_loss"] = cranked_loss
        elif lossName == "cross_entropy_loss":
            if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                cross_entropy_Labelsmooth_loss = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
                lossClasses["cross_entropy_loss"] = cross_entropy_Labelsmooth_loss
            else:
                if cfg.MODEL.CLASSIFIER_NAME != "hierarchy_linear":
                    cross_entropy_loss = CrossEntropyLoss()
                else:
                    cross_entropy_loss = NLLLoss()
                lossClasses["cross_entropy_loss"] = cross_entropy_loss
        elif lossName == "cluster_loss":
            cluster_loss = ClusterLoss(num_classes=6, feat_dim=2048, r_outer=5, io_ratio=4, distance_type="cos", use_gpu=True)
            lossClasses["cluster_loss"] = cluster_loss
        elif lossName == "one_vs_rest_loss":
            one_vs_rest_loss = OneVsRestLoss() #feat, label, normalize_feature=False)
            lossClasses["one_vs_rest_loss"] = one_vs_rest_loss
        elif lossName == "attention_loss":
            attention_loss = AttentionLoss(2, 2, 0.4)
            lossClasses["attention_loss"] = attention_loss
        elif lossName == "class_predict_loss":
            class_predict_loss = ClassPredictLoss()
            lossClasses["class_predict_loss"] = class_predict_loss
        elif lossName == "margin_loss":
            margin_loss = MarginLoss()
            lossClasses["margin_loss"] = margin_loss
        elif lossName == "multilabel_binary_cross_entropy_loss":
            multilabel_binary_cross_entropy_loss = MultilabelBinaryCrossEntropy()
            lossClasses["multilabel_binary_cross_entropy_loss"] = multilabel_binary_cross_entropy_loss
        elif lossName == "seg_mask_loss":
            seg_mask_loss = SegMaskLoss(cfg.MODEL.SEG_NUM_CLASSES)
            lossClasses["seg_mask_loss"] = seg_mask_loss
        elif lossName == "gcam_mask_loss":
            gcam_mask_loss = GradCamMaskLoss(cfg.MODEL.SEG_NUM_CLASSES)
            lossClasses["gcam_mask_loss"] = gcam_mask_loss
        elif lossName == "pos_masked_img_loss":
            pos_masked_img_loss = PosMaskedImgLoss()
            lossClasses["pos_masked_img_loss"] = pos_masked_img_loss
        elif lossName == "neg_masked_img_loss":
            neg_masked_img_loss = NegMaskedImgLoss()
            lossClasses["neg_masked_img_loss"] = neg_masked_img_loss
        elif lossName == "mse_loss":
            mse_loss = MSELoss()
            lossClasses["mse_loss"] = mse_loss
        elif lossName == "for_show_loss":
            for_show_loss = ForShowLoss()
            lossClasses["for_show_loss"] = for_show_loss

        else:
            raise Exception('expected METRIC_LOSS_TYPE should be similarity_loss, ranked_loss, cranked_loss'
              'but got {}'.format(cfg.LOSS.TYPE))

    """
    if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
        print("label smooth on, numclasses:", num_classes)
    """

    #计算loss的函数
    def D_loss_func(feat=None, logit=None, label=None, regression_logit=None, regression_label=None, feat_attention=None, similarity=None, similarity_label=None, multilabel=None, seg_mask=None, seg_gtmask=None, seg_label=None, gcam_mask=None, gcam_gtmask=None, gcam_label=None, origin_logit=None, pos_masked_logit=None, neg_masked_logit=None, show=0):
        losses = {}
        for lossName in lossKeys:
            if lossName == "similarity_loss":
                losses["similarity_loss"] = similarity_loss(feat_attention, label, multilabel)#(similarity, similarity_label)
            elif lossName == "ranked_loss":
                losses["ranked_loss"] = ranked_loss(feat, label, normalize_feature=True)  # ranked_loss
            elif lossName == "cranked_loss":
                losses["cranked_loss"] = cranked_loss(feat, label)  # cranked_loss
            elif lossName == "cross_entropy_loss":
                if cfg.MODEL.CLASSIFIER_NAME != "hierarchy_linear":
                    losses["cross_entropy_loss"] = lossClasses["cross_entropy_loss"](logit, label)
                else:
                    losses["cross_entropy_loss"] = lossClasses["cross_entropy_loss"](torch.log(logit), label)
            elif lossName == "cluster_loss":
                losses["cluster_loss"] = cluster_loss(feat, label)
            elif lossName == "one_vs_rest_loss":
                losses["one_vs_rest_loss"] = one_vs_rest_loss(logit, label)
            elif lossName == "attention_loss":
                losses["attention_loss"] = attention_loss(feat_attention, label, normalize_feature=True)
            elif lossName == "class_predict_loss":
                losses["class_predict_loss"] = class_predict_loss(logit, label)
            elif lossName == "margin_loss":
                losses["margin_loss"] = margin_loss(logit, similarity_label)
            elif lossName == "multilabel_binary_cross_entropy_loss":
                losses["multilabel_binary_cross_entropy_loss"] = multilabel_binary_cross_entropy_loss(logit, multilabel)
            elif lossName == "seg_mask_loss":
                losses["seg_mask_loss"] = seg_mask_loss(seg_mask, seg_gtmask, seg_label)
            elif lossName == "gcam_mask_loss":
                losses["gcam_mask_loss"] = gcam_mask_loss(gcam_mask, gcam_gtmask, gcam_label)
            elif lossName == "pos_masked_img_loss":
                losses["pos_masked_img_loss"] = pos_masked_img_loss(pos_masked_logit, neg_masked_logit, origin_logit, label)
            elif lossName == "neg_masked_img_loss":
                losses["neg_masked_img_loss"] = neg_masked_img_loss(pos_masked_logit, neg_masked_logit, origin_logit, label)
            elif lossName == "mse_loss":
                losses["mse_loss"] = mse_loss(regression_logit, regression_label)
            elif lossName == "for_show_loss":
                losses["for_show_loss"] = for_show_loss(show=show)
            else:
                raise Exception('expected METRIC_LOSS_TYPE should be similarity_loss, ranked_loss, cranked_loss'
                                'but got {}'.format(cfg.LOSS.TYPE))
        return losses

    """
    if sampler == 'softmax':
        def G_loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'ranked_loss':
        def G_loss_func(score, feat, target):
            # return ranked_loss(feat, target)[0]
            return ranked_loss(feat, target, normalize_feature=False)
    elif cfg.DATA.DATALOADER.SAMPLER == 'cranked_loss':
        def G_loss_func(score, feat, target):
            # return cranked_loss(feat, target)[0]
            return cranked_loss(feat, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'softmax_rank':
        def G_loss_func(score, feat, target):
            if cfg.LOSS.TYPE == 'ranked_loss':
                if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                    # return  xent(score, target) + cfg.SOLVER.WEIGHT*ranked_loss(feat, target)[0] # new add by zzg, open label smooth
                    return xent(score, target) + cfg.LOSS.WEIGHT * ranked_loss(feat,
                                                                               target, normalize_feature=False)  # CJY at 2019.9.23, 这个改动与pytorch版本有关
                else:
                    # return F.cross_entropy(score, target) + ranked_loss(feat, target)[0]    # new add by zzg, no label smooth
                    return F.cross_entropy(score, target) + ranked_loss(feat,
                                                                        target, normalize_feature=False)  # CJY at 2019.9.23, 这个改动与pytorch版本有关

            elif cfg.LOSS.TYPE == 'cranked_loss':
                if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                    # return  xent(score, target) +cfg.SOLVER.WEIGHT*cranked_loss(feat, target)[0] # new add by zzg, open label smooth
                    return xent(score, target) + cfg.LOSS.WEIGHT * cranked_loss(feat,
                                                                                target)  # CJY at 2019.9.23, 这个改动与pytorch版本有关
                else:
                    # return F.cross_entropy(score, target) + cranked_loss(feat, target)[0]    # new add by zzg, no label smooth
                    return F.cross_entropy(score, target) + cranked_loss(feat,
                                                                         target)  # CJY at 2019.9.23, 这个改动与pytorch版本有关
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.LOSSL.TYPE))
    else:
        print('expected sampler should be softmax, ranked_loss or cranked_loss, '
              'but got {}'.format(cfg.DATA.DATALOADER.SAMPLER))
    """

    return D_loss_func, lossClasses

def make_G_loss(cfg, num_classes):   #注意是找相同点common
    """
    common_loss = CommonLoss(num_classes=num_classes, margin=cfg.LOSS.MARGIN_RANK)  # common_loss
    def D_loss_func(score, feat, target):
        return common_loss(score, target)
    """
    """
    if cfg.LOSS.TYPE == 'ranked_loss':
        common_loss = CommonLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # ranked_loss
    def D_loss_func(score, feat, target):
        return common_loss(feat, target, normalize_feature=False)
    """
    """
    common_loss = CommonLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # ranked_loss
    similarity_loss = SimilarityLoss()
    def G_loss_func(feat, score, label, similarity, similarity_label):
        return [F.cross_entropy(score, label), similarity_loss(similarity, similarity_label)]
    """
    class_predict_loss = ClassPredictLoss()
    def G_loss_func(feat, score, label):
        return  class_predict_loss(score, label)
    return G_loss_func

