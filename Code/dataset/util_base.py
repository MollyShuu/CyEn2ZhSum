'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：Pytorch -> util_base.py
@Author ：MollyShuu
@Date   ：2021/4/21 17:12
@IDE    ：PyCharm
==================================================
'''
# （三）util_base.py -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
from dataset.dataset_base import TranslationDataset
from dataset.filed_base import Field
from Utils import get_weight2


def build_dataset(opt, data_path, vocab_path, device, prob_and_idx, vocab_src, vocab_tgt, train=True):
    src = data_path[0]
    tgt = data_path[1]

    src_field = Field(unk=True, pad=True, bos=False, eos=False)
    tgt_field = Field(unk=True, pad=True, bos=True, eos=True)

    if (opt.emd_type == 1):
        if len(vocab_path) == 1:
            # use shared vocab
            src_vocab = tgt_vocab = vocab_path[0]
            src_special = tgt_special = sorted(set(src_field.special + tgt_field.special))
        else:
            src_vocab, tgt_vocab = vocab_path
            src_special = src_field.special
            tgt_special = tgt_field.special
        with open(src_vocab, encoding="utf-8-sig") as f:
            src_words = [line.strip() for line in f]
        with open(tgt_vocab, encoding="utf-8-sig") as f:
            tgt_words = [line.strip() for line in f]
        weight_src = 0
        weight_tgt = 0
    if (opt.emd_type == 2):
        if len(vocab_path) == 1:
            # use shared vocab
            src_vocab = tgt_vocab = vocab_path[0]
            src_special = tgt_special = []
        else:
            src_special = []
            tgt_special = []
        # vocab_src = KeyedVectors.load_word2vec_format(opt.vocab[0])
        # vocab_tgt = KeyedVectors.load_word2vec_format(opt.vocab[1])
        if train:
            weight_src = 1
            weight_tgt = 1
        else:
            weight_src = get_weight2(vocab_src, opt, device)
            weight_tgt = get_weight2(vocab_tgt, opt, device)

        src_words = vocab_src.index2word
        tgt_words = vocab_tgt.index2word
    # 词典
    src_field.load_vocab(src_words, src_special)
    tgt_field.load_vocab(tgt_words, tgt_special)
    src_field.load_trans_prob(prob_and_idx)

    return TranslationDataset(src, tgt, opt, device, train, {'src': src_field, 'tgt': tgt_field}), {
        'src_weight': weight_src, 'tgt_weight': weight_tgt}
