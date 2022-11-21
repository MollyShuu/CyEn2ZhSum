'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：Pytorch -> filed_base.py
@Author ：MollyShuu
@Date   ：2021/4/21 17:06
@IDE    ：PyCharm
==================================================
'''
# （一）filed_base.py -*- coding: utf-8 -*-
# typing类型检查，防止运行时出现参数和返回值类型不符合；不报错只提醒https://www.cnblogs.com/cwp-bg/p/7825729.html
# 1为enc_inputs,2为dec_inputs,3为dec_outputs
from typing import List
import numpy as np
import torch

# 添加特殊token，保证模型把它拆分
#  <pad>:将句子补充至最长的长度；<bos>：标识句子开始；<eos>：标识句子结束；<unk>：语料库中未出现的词，称做“未登录词”
EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


# field 类似于return的返回迭代器
class Field(object):
    def __init__(self, bos: bool, eos: bool, pad: bool, unk: bool):
        self.bos_token = BOS_TOKEN if bos else None
        self.eos_token = EOS_TOKEN if eos else None
        self.unk_token = UNK_TOKEN if unk else None
        self.pad_token = PAD_TOKEN if pad else None

        self.vocab = None

    def load_vocab(self, words: List[str], specials: List[str]):  # 参数名words：类型list[str]
        self.vocab = Vocab(words, specials)

    def load_trans_prob(self, prob_and_idx):
        self.problist, self.idxlist = prob_and_idx

    # -----将一个batch进行编码--------#
    # 这里enc_inputs,dec_inputs,dec_outputs 应分开编码
    def process(self, batch, device, max_swords, prc_type):
        '''
        prc_type:1/4为enc_inputs(本项目使用4),2为dec_inputs,3为dec_outputs
        '''
        #         for x in batch:
        #             print('========Filed:',len(x),x)
        max_len = max(len(x) for x in batch)
        max_len = min(max_len, 200)  # tgt句子最大长度
        padded = []
        if (prc_type == 1):
            Gsen_feature_padded, Gsen_adj_padded, Gcpt_feature_padded, Gcpt_adj_padded = self.pad_batch(batch,
                                                                                                        max_swords)
            return Gsen_feature_padded.long().to(device), Gsen_adj_padded.long().to(
                device), Gcpt_feature_padded.long().to(device)  # Gcpt_adj_padded.long().to(device)
        elif (prc_type == 4):
            Gsen_feature_padded, Gsen_adj_padded = self.pad_sen_batch(batch, max_swords)
            Gcpt_feature_padded, probmatrix, idxmatrix = self.pad_ctp(batch)
            return Gsen_feature_padded.long().to(device), Gsen_adj_padded.long().to(
                device), Gcpt_feature_padded.long().to(device), probmatrix.float().to(device), idxmatrix.long().to(
                device)
        elif (prc_type == 2):
            for x in batch:
                bos = [self.bos_token]
                eos = [self.eos_token]
                if (len(x) <= max_len):
                    pad = [self.pad_token] * (max_len - len(x))
                    padded.append(bos + x + eos + pad)
                else:
                    padded.append(bos + x[:max_len] + eos)
            padded = torch.tensor([self.tgt_encode(ex) for ex in padded])
            # return padded.long()
            return padded.long().to(device)
        elif (prc_type == 3):
            for x in batch:
                bos = [self.bos_token]
                eos = [self.eos_token]
                if (len(x) <= max_len):
                    pad = [self.pad_token] * (max_len - len(x) + 1)
                    padded.append(x + eos + pad)
                else:
                    pad = [self.pad_token]
                    padded.append(x[:max_len] + eos + pad)
            padded = torch.tensor([self.tgt_encode(ex) for ex in padded])
            # return padded.long()
            return padded.long().to(device)

    def pad_ctp(self, batch):
        concepts_batch = [x.Gcpt_vertex_features for x in batch]
        max_len = max(len(x) for x in concepts_batch)
        max_len = min(max_len, 200)
        padded = []
        for x in concepts_batch:
            if (len(x) < max_len):
                pad = [self.pad_token] * (max_len - len(x))
                padded.append(x + pad)
            else:
                padded.append(x[:max_len])
        # print(padded)
        token_list, prob_list, idx_list = [], [], []
        for ex in padded:
            tokens, probs, idxes = self.cpt_encode(ex)
            token_list.append(tokens)
            prob_list.append(probs)
            idx_list.append(idxes)
        padded = torch.tensor(token_list)
        probmatrix = torch.tensor(prob_list)
        idxmatrix = torch.tensor(idx_list)
        return padded, probmatrix, idxmatrix

    def pad_vfeatures(self, Gsen_vertex_features, max_len, max_nodes, pad_token='<pad>'):
        '''
        对1个batch里的G进行pad，每个graph的N_nodes,N_sen_words数都应该一样->[N_nodes,N_words],同时邻接矩阵也要进行同样的pad
        '''
        padded = []
        for x in Gsen_vertex_features:
            if (len(x) >= max_len):
                padded.append(x[:max_len])
            else:
                pad = [pad_token] * (max_len - len(x))
                padded.append(x + pad)

        if len(Gsen_vertex_features) > max_nodes:
            padded = padded[:max_nodes]
        else:
            [padded.append([pad_token] * max_len) for i in range(max_nodes - len(Gsen_vertex_features))]

        #[padded.append([pad_token] * max_len) for i in range(max_nodes - len(Gsen_vertex_features))]
        padded = [self.src_encode(ex) for ex in padded]
        return padded

    def pad_sen_batch(self, batch, max_swords):
        # (1)对Gsen进行pad
        max_nodes = max(len(x.Gsen_vertex_features) for x in batch)
        max_nodes = min(max_nodes, 150)
        max_len = max_swords
        # max_len = max(max(len(sen) for sen in x.Gcpt_vertex_features) for x in batch)
        # max_len = min(max_len, max_swords)
        Gsen_feature_padded = [self.pad_vfeatures(x.Gsen_vertex_features, max_len, max_nodes) for x in batch]
        Gsen_adj_padded = []
        for x in batch:
            if len(x.Gsen_adj_martix) > max_nodes:
                print(max_nodes, len(x.Gsen_adj_martix)) # N*N
                Gsen_adj_padded.append([x.Gsen_adj_martix[i][:max_nodes] for i in range(max_nodes)])
            else:
                Gsen_adj_padded.append(np.pad(x.Gsen_adj_martix,
                                              ((0, max_nodes - len(x.Gsen_adj_martix)),
                                               (0, max_nodes - len(x.Gsen_adj_martix))),
                                              'constant', constant_values=(0, 0)))


        # Gsen_adj_padded = [np.pad(x.Gsen_adj_martix,
        #                           ((0, max_nodes - len(x.Gsen_adj_martix)), (0, max_nodes - len(x.Gsen_adj_martix))),
        #                           'constant', constant_values=(0, 0)) for x in batch]
        # print(Gsen_adj_padded)
        return torch.tensor(Gsen_feature_padded), torch.tensor(Gsen_adj_padded)

    def pad_batch(self, batch, max_swords, max_cwords=6):
        # (1)对Gsen进行pad
        max_nodes = max(len(x.Gsen_vertex_features) for x in batch)
        max_len = max(max(len(sen) for sen in x.Gcpt_vertex_features) for x in batch)
        max_len = min(max_len, max_swords)
        # max_len = max((max(len(sen) for sen in x.Gsen_vertex_features)) for x in batch)
        Gsen_feature_padded = [self.pad_vfeatures(x.Gsen_vertex_features, max_len, max_nodes) for x in batch]
        Gsen_adj_padded = [np.pad(x.Gsen_adj_martix,
                                  ((0, max_nodes - len(x.Gsen_adj_martix)), (0, max_nodes - len(x.Gsen_adj_martix))),
                                  'constant', constant_values=(0, 0)) for x in batch]
        # （2）对Gcpt进行pad
        max_nodes = max(len(x.Gcpt_vertex_features) for x in batch)
        max_len = max(max(len(sen) for sen in x.Gcpt_vertex_features) for x in batch)
        max_len = min(max_len, max_cwords)
        # max_len = max((max(len(sen) for sen in x.Gcpt_vertex_features)) for x in batch)
        Gcpt_feature_padded = [self.pad_vfeatures(x.Gcpt_vertex_features, max_len, max_nodes) for x in batch]
        Gcpt_adj_padded = [np.pad(x.Gcpt_adj_martix,
                                  ((0, max_nodes - len(x.Gcpt_adj_martix)), (0, max_nodes - len(x.Gcpt_adj_martix))),
                                  'constant', constant_values=(0, 0)) for x in batch]
        return torch.tensor(Gsen_feature_padded), torch.tensor(Gsen_adj_padded), torch.tensor(
            Gcpt_feature_padded), torch.tensor(Gcpt_adj_padded)

    def prb_idx(self, batch, device):
        '''
        ‘the’ 对应的每个概率为0，可以用来补全
        :param batch:
        :param device:
        :return:
        '''
        probs, idxes = [], []
        pad = [0.0]
        max_len = max([len(concepts) for concepts in batch])
        max_words = max([max([len(toks) for toks in concepts]) for concepts in batch])
        for concepts in batch:
            prob, idx = [], []
            for concept in concepts:
                probt, idxt = [], []
                for tok in concept:
                    if tok in self.vocab.stoi:
                        tokidx = self.vocab.stoi[tok]
                    else:
                        tokidx = self.unk_id
                    probt.append(self.problist[tokidx][0])  # 保存token对应的tgt的概率
                    idxt.append(self.idxlist[tokidx][0])  # 保存token对应的tgt的id
                # --------补齐每个concepts节点的长度--------#
                prob.append(probt + pad * (max_words - len(probt)))
                idx.append(idxt + pad * (max_words - len(idxt)))
            # -------------batch补全对齐-----------------#
            prob += [pad * max_words] * (max_len - len(concepts))
            idx += [pad * max_words] * (max_len - len(concepts))
            probs.append(prob)
            idxes.append(idx)
        probmatrix = torch.tensor(probs)
        idxmatrix = torch.tensor(idxes)
        # return probmatrix.float(), idxmatrix.long()
        return probmatrix.float().to(device), idxmatrix.long().to(device)

    def prb_idx_words(self, batch, device):
        '''
        ‘the’ 对应的每个概率为0，可以用来补全
        :param batch:
        :param device:
        :return:
        '''
        probs, idxes = [], []
        pad = [[0.0, 0.0, 0.0]]
        max_len = max([len(concepts) for concepts in batch])
        for concepts in batch:
            concepts = list(set(concepts))
            prob, idx = [], []
            for tok in concepts:
                if tok in self.vocab.stoi:
                    tokidx = self.vocab.stoi[tok]
                else:
                    tokidx = self.unk_id
                prob.append(self.problist[tokidx])  # 保存token对应的tgt的概率
                idx.append(self.idxlist[tokidx])  # 保存token对应的tgt的id
            # -------------batch补全对齐-----------------#
            prob += (pad * (max_len - len(concepts)))
            idx += (pad * (max_len - len(concepts)))
            probs.append(prob)
            idxes.append(idx)
        probmatrix = torch.tensor(probs)
        idxmatrix = torch.tensor(idxes)
        return probmatrix.float().to(device), idxmatrix.long().to(device)

    def cpt_encode(self, tokens):
        ids, prob, idx = [], [], []
        for tok in tokens:
            if tok in self.vocab.stoi:
                tokidx = self.vocab.stoi[tok]
            else:
                tokidx = self.unk_id
            prob.append(self.problist[tokidx])
            idx.append(self.idxlist[tokidx])
            ids.append(tokidx)
        return ids, prob, idx

    def src_encode(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.vocab.stoi:
                tokidx = self.vocab.stoi[tok]
            else:
                tokidx = self.unk_id
            ids.append(tokidx)  # 保存token本身的id/index
        return ids

    def tgt_encode(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.vocab.stoi:
                ids.append(self.vocab.stoi[tok])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):
        tokens = []
        for tok in ids:
            tok = self.vocab.itos[tok]
            if tok == self.eos_token:
                break
            if tok == self.bos_token:
                continue
            tokens.append(tok)
        # 删除BPE符号，按照T2T切分-。
        return " ".join(tokens).replace("@@ ", "").replace("@@", "").replace("-", " - ")

    @property
    def special(self):
        return [tok for tok in [self.unk_token, self.pad_token, self.bos_token, self.eos_token] if tok is not None]

    @property
    def pad_id(self):
        return self.vocab.stoi[self.pad_token]  # 返回每个词和其相应的编号

    @property
    def eos_id(self):
        return self.vocab.stoi[self.eos_token]

    @property
    def bos_id(self):
        return self.vocab.stoi[self.bos_token]

    @property
    def unk_id(self):
        return self.vocab.stoi[self.unk_token]


# 将词典进行编码成向量/数字（one-hot），词典是去重后的
class Vocab(object):
    def __init__(self, words: List[str], specials: List[str]):
        self.itos = specials + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}  # 单词和其对应的编号

    def __len__(self):
        return len(self.itos)
