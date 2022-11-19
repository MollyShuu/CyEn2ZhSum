'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> encoder
@Author ：MollyShuu
@Date   ：2021/6/1 11:55
@IDE    ：PyCharm
==================================================
'''
import torch
import torch.nn as nn
from layers.attention import PoswiseFeedForwardNet, PositionalEncoding, MultiHeadAttention, get_attn_pad_mask


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, ff_size):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(embedding_dim,  heads)  
        self.pos_ffn = PoswiseFeedForwardNet(embedding_dim, ff_size)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, opt, src_vocab_size, padding_idx, weight):
        super(Encoder, self).__init__()
        self.embedding_dim = opt.embedding_dim
        self.padding_idx = padding_idx
        self.src_emb = nn.Embedding(src_vocab_size, opt.embedding_dim, padding_idx)
        self.reset_parameters()

        self.layers = nn.ModuleList([EncoderLayer(opt.embedding_dim, opt.heads, opt.ff_size) for _ in
                                     range(opt.layers)])  
    def reset_parameters(self):
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=self.embedding_dim ** -0.5)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs,
                                               mask_pad=self.padding_idx) 
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
