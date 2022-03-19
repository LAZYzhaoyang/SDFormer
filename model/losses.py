"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .Blocks import *
import numpy as np

#============================Loss Function============================#

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        
        
        
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            
            target = target.view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2))
        #target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1
        
        #probs = F.sigmoid(logits)
        #m1 = probs.view(num, -1)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class mIOULoss(nn.Module):
    def __init__(self, n_classes, size_average=True):
        super(mIOULoss, self).__init__()
        self.n_classes = n_classes
 
    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1
        
        #probs = F.sigmoid(logits)
        #m1 = probs.view(num, -1)
        m1 = logits.view(num, self.n_classes, -1)
        m2 = targets.view(num, self.n_classes, -1)
        intersection = (m1 * m2)

        score = (smooth + intersection.sum(2))/(smooth + m1.sum(2) + m2.sum(2) - intersection.sum(2))
        score = 1 - score.sum(1)/self.n_classes
        score = score.sum()/num
 
        #score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        #score = 1 - score.sum() / num
        return score

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
 
    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1
        
        #probs = F.sigmoid(logits)
        #m1 = probs.view(num, -1)
        m1 = logits.view(num, self.n_classes, -1)
        m2 = targets.view(num, self.n_classes, -1)
        intersection = (m1 * m2)

        score = (smooth + 2*intersection.sum(2))/(smooth + m1.sum(2) + m2.sum(2))
        score = 1 - score.sum(1)/self.n_classes
        score = score.sum()/num

        return score


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, class_weight = None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        self.class_weight =class_weight
        
        if self.class_weight is None:
            self.class_weight = torch.ones(self.num_class, 1)
        elif isinstance(self.class_weight, (list, np.ndarray)):
            assert len(self.class_weight) == self.num_class
            self.class_weight = torch.FloatTensor(class_weight).view(self.num_class, 1)
            #self.class_weight = self.class_weight / self.class_weight.sum()
        elif isinstance(self.class_weight, float):
            weight = torch.ones(self.num_class, 1)
            weight = weight * (1 - self.class_weight)
            weight[balance_index] = self.class_weight
            self.class_weight = weight
        else:
            raise TypeError('Not support alpha type')

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, pred, target):
        if self.smooth:
            target = torch.clamp(target, self.smooth, 1.0-self.smooth)
        if pred.dim()>2:
            pred = pred.view(pred.size(0), pred.size(1), -1)
        if target.dim()==4:
            target = target.view(target.size(0), target.size(1), -1)
        elif not pred.size()==target.size():
            raise ValueError('pred.size should be same as target.size')
        epsilon = 1e-10
        alpha = self.alpha
        gamma = self.gamma
        class_weight = self.class_weight
        
        if alpha.device != target.device:
            alpha = alpha.to(target.device)
        if class_weight.device != target.device:
            class_weight = class_weight.to(target.device)
        if pred.device != target.device:
            pred = pred.to(target.device)
        
        loss = - alpha * target * ((1.0 - pred) ** gamma) * torch.log(pred) \
            - (1.0 - alpha) * (1 - target) * (pred ** gamma) * torch.log(1 - pred)
        loss = class_weight * loss
        #loss = loss.sum(2)
        #logit = F.softmax(input, dim=1)
        """
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
        
        
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        """
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class JointLoss(nn.Module):
    def __init__(self, firstloss, secondloss, first_weight=0.5):
        super(JointLoss, self).__init__()
        """
        if not type(firstloss)==nn.Module:
            raise TypeError('firstloss should be nn.Module')
        if not type(secondloss)==nn.Module:
            raise TypeError('secondloss should be nn.Module')
            """
        if first_weight < 0 or first_weight > 1.0:
            raise ValueError('firstweight value should be in [0,1]')
        second_weight = 1 - first_weight
        self.firstloss = firstloss
        self.secondloss = secondloss
        self.firstweight = first_weight
        self.secondweight = second_weight
        
    def forward(self, pred, target):
        loss1 = self.firstloss(pred, target)
        loss2 = self.secondloss(pred, target)
        
        loss = self.firstweight*loss1 + self.secondweight*loss2
        return loss
    