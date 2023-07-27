# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
# poisonous_lvl = pd.read_csv(
#     "http://ptak.felk.cvut.cz/plants//DanishFungiDataset/poison_status_list.csv"
# )
poisonous_lvl = pd.read_csv('datasets/fungi/challenge_data/poison_status_list.csv')
POISONOUS_SPECIES = poisonous_lvl[poisonous_lvl["poisonous"] == 1].class_id.unique()

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def seesaw_ce_loss(cls_score,
                   labels,
                   label_weights,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   eps,
                   reduction='mean',
                   avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.
    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        label_weights (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor，各个类别的样本数量
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)      # sample_ratio_matrix[i][j]表示第i个类别的样本数占第j个类别的样本数的比例
        index = (sample_ratio_matrix < 1.0).float()         # 小于1的元素设为1
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor # 第2个样本：[0.5, 1, 1, 1, 1]，第3个样本：[0.33, 0.67, 1, 1, 1]

    # compensation factor，当前类别的分数作为基准
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)     # q=2，分数过高的类别设为1，分数过小的进一步抑制
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))    # 少样本类别、分数过低的类别，加一个负数进一步抑制

    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')

    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(
        loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss


class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    arXiv:
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
             of softmax. Only False is supported.
        p (float, optional): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float, optional): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int, optional): The number of classes.
             Default to 1203 for LVIS v1 dataset.
        eps (float, optional): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method that reduces the loss to a
             scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        return_dict (bool, optional): Whether return the losses as a dict.
             Default to True.
    """

    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1604,
                 eps=1e-2,
                 reduction='mean',
                 loss_weight=1.0,
                 return_dict=False):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_dict = return_dict

        # 0 for pos, 1 for neg
        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_classes + 1, dtype=torch.float))

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

    def forward(self,
                cls_score,
                labels,
                label_weights=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.
            label_weights (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor | Dict [str, torch.Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes
        pos_inds = torch.logical_and(labels < self.num_classes, labels >= 0)
        nov_inds = labels == -1
        if self.num_classes > 1000:
            poison_species =  torch.from_numpy(POISONOUS_SPECIES).to(device=labels.device)
            poison_inds = (labels.unsqueeze(1) == poison_species.unsqueeze(0)).sum(1) >= 1
        else:
            poison_inds = torch.tensor([0])
        # poison_inds = [i for i, label in enumerate(labels) if label in POISONOUS_SPECIES]
        # 0 for pos, 1 for neg
        # obj_labels = (labels == self.num_classes).long()

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)

        # calculate loss_cls_classes (only need pos samples)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score[pos_inds], labels[pos_inds],
                label_weights[pos_inds], self.cum_samples[:self.num_classes],
                self.num_classes, self.p, self.q, self.eps, reduction,
                avg_factor)
        else:
            loss_cls_classes = cls_score[pos_inds].sum()

        if nov_inds.sum() > 0:
            cls_score_nov = cls_score[nov_inds]
            labels_nov = torch.ones_like(cls_score_nov) / cls_score_nov.shape[1]
            loss_cls_classes_nov = - (labels_nov * F.log_softmax(cls_score_nov, dim=1)).sum(dim=-1)
            loss_cls_classes_nov = self.loss_weight * loss_cls_classes_nov.mean()
        else:
            loss_cls_classes_nov = torch.tensor(0.)

        # poisonous classification
        if poison_inds.sum() > 0:
            poison_mask = torch.zeros(cls_score.shape[-1], device=cls_score.device)
            poison_mask[poison_species] = 1

            cls_score_poison = cls_score[poison_inds][:, poison_mask == 1]
            topk_scores, _ = torch.topk(cls_score_poison, k=5, dim=1)
            cls_score_poison = topk_scores.mean(1)
            cls_score_edible = cls_score[poison_inds][:, poison_mask != 1]
            topk_scores, _ = torch.topk(cls_score_edible, k=5, dim=1)
            cls_score_edible = topk_scores.mean(1)

            cls_score_poi = torch.stack([cls_score_poison, cls_score_edible], dim=1)
            loss_cls_classes_poison = - F.log_softmax(cls_score_poi, dim=1)[:, 0]
            loss_cls_classes_poison = self.loss_weight * loss_cls_classes_poison.mean()
        else:
            loss_cls_classes_poison = torch.tensor(0.)

        loss_cls_classes = (loss_cls_classes * pos_inds.sum() + loss_cls_classes_nov * nov_inds.sum() + loss_cls_classes_poison * poison_inds.sum()) \
                           / (pos_inds.sum() + nov_inds.sum() + poison_inds.sum())

        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls'] = loss_cls_classes
        else:
            loss_cls = loss_cls_classes

        return loss_cls
