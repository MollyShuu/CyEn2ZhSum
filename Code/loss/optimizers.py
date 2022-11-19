'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> optimizers
@Author ：MollyShuu
@Date   ：2021/4/22 15:54
@IDE    ：PyCharm
==================================================
'''
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import dot, linalg


class WarmAdam(object):
    def __init__(self, params, lr, hidden_size, warm_up, n_step):
        self.original_lr = lr
        self.n_step = n_step
        self.hidden_size = hidden_size
        self.warm_up_step = warm_up  # 超过该值后学习率开始下降
        self.optimizer = optim.Adam(params, betas=[0.9, 0.998], eps=1e-8)

    #  self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, eta_min=1e-8, last_epoch=-1)

    def step(self):
        self.n_step = self.n_step + 1
        warm_up = min(self.n_step ** (-0.5), self.n_step * self.warm_up_step ** (-1.5))
        lr = self.original_lr * (self.hidden_size ** (-0.5) * warm_up)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    # # -----采用余弦退火学习率----#
    # def step(self):
    #     self.n_step = self.n_step + 1
    #     self.optimizer.step()
    #     self.scheduler.step()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index):
        self.padding_idx = ignore_index
        self.label_smoothing = label_smoothing  # 0.1
        self.vocab_size = tgt_vocab_size

        super(LabelSmoothingLoss, self).__init__()

    # 如果要改，生成改成词向量，dec_outputs也应该改成词向量
    def forward(self, output, target):
        target = target[:, 1:].contiguous().view(-1)  # 展开铺平成1维，取个1：是因为在NLCS中tgt开头为<bos>
        output = output.view(-1, self.vocab_size)  # size [248]
        # output = output[:, 0:-1].contiguous().view(-1, self.vocab_size)  # size [248, 5003]
        non_pad_mask = target.ne(self.padding_idx)  # .ne函数，逐一比较是否‘不相同’
        nll_loss = -output.gather(dim=-1, index=target.view(-1, 1))[
            non_pad_mask].sum()  # [248,1],按index在dim维度上取值，输出和index维度相同
        smooth_loss = -output.sum(dim=-1, keepdim=True)[non_pad_mask].sum()
        eps_i = self.label_smoothing / self.vocab_size
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        return loss / non_pad_mask.float().sum()

class MSELoss(nn.Module):
    def __init__(self, ignore_vec):
        super(MSELoss, self).__init__()
        self.padding_vec = ignore_vec
        self.linear = nn.Linear(1, 1, bias=False)  # loss中的λ参数，自我学习

    def forward(self, output, target):
        target = target.view(-1, target.size(-1))  # [batch_size*tgt_len,embedding_dim]
        output = output.view(-1, target.size(-1))  # [batch_size*tgt_len,embedding_dim]
        non_pad_mask = target.ne(self.padding_vec)  # torch.tensor(pad_vecs["tgt"]),判断是否不相同
        non_pad_mask = [True in vec for vec in non_pad_mask]  # [batch_size*tgt_len], 只要有1个维度不相同则不为pad
        loss = torch.mean((output - target).abs()[non_pad_mask].sum(dim=1) ** 2)  # MSELoss = (x-y)^2
        # 从语义上，应该加上整句相似的惩罚
        return loss


class SmoothL1Loss(nn.Module):
    # SmoothL1Loss小于MSELoss
    # 主要用于计算目标检测（如Fast-RCNN）中的位置偏差损失。相比于L1Loss，SmoothL1Loss对于差异在某个范围（比如1）之内的惩罚要更小，表明网络侧重于位置偏差过大的点，这主要是因为在目标检测网络中真实位置标签也可能存在一定的误差，而且评估标准采用的IOU门限一般为0.5，因此在一定范围内可以放宽要求。
    def __init__(self, ignore_vec, sigma=1.0, reduce=False):
        # https://blog.csdn.net/weixin_43593330/article/details/108165617
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.padding_vec = ignore_vec
        # self.linear = nn.Linear(1, 1, bias=False)  # loss中的λ参数，自我学习

    def forward(self, output, target):
        target = target.view(-1, target.size(-1))  # [batch_size*tgt_len,embedding_dim]
        output = output.view(-1, target.size(-1))  # [batch_size*tgt_len,embedding_dim]
        non_pad_mask = target.ne(self.padding_vec)  # torch.tensor(pad_vecs["tgt"]),判断是否不相同
        non_pad_mask = [True in vec for vec in non_pad_mask]  # [batch_size*tgt_len], 只要有1个维度不相同则不为pad
        beta = 1. / (self.sigma ** 2)
        diff = (output - target).abs()[non_pad_mask].sum(dim=1)  # non_pad_mask里为false被删除了
        cond = diff < beta
        loss_word = torch.mean(torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta))  # SmoothL1Loss
        loss_sen = sentence_simi(output, target)
        # 从语义上，应该加上整句相似的惩罚
        loss = loss_word - loss_sen
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, ignore_vec):
        super(CosineSimilarityLoss, self).__init__()
        self.padding_vec = ignore_vec
        self.linear = nn.Linear(1, 1, bias=False)  # loss中的λ参数，自我学习
        self.loss_fct = nn.MSELoss()

    def forward(self, output, target):
        target = target.view(-1, target.size(-1))  # [batch_size*tgt_len,embedding_dim]
        output = output.view(-1, target.size(-1))  # [batch_size*tgt_len,embedding_dim]
        non_pad_mask = target.ne(self.padding_vec)  # torch.tensor(pad_vecs["tgt"])
        non_pad_mask = [True in vec for vec in non_pad_mask]  # [batch_size*tgt_len]
        simi = torch.cosine_similarity(target, output, dim=1)
        label = torch.ones([len([non_pad_mask])])
        loss = self.loss_fct(simi[non_pad_mask], label)
        print(loss)


def sentence_simi(output, target):
    sentence_tgt = target.sum(dim=0) / target.size()[0]
    sentence_out = output.sum(dim=0) / output.size()[0]
    similarity = dot(sentence_out, sentence_tgt) / (linalg.norm(sentence_out) * linalg.norm(sentence_tgt))
    return similarity
