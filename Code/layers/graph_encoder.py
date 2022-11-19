'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> graph_encoder
@Author ：MollyShuu
@Date   ：2021/5/10 9:51
@IDE    ：PyCharm
==================================================
'''

import torch
import torch.nn as nn
from layers.gat import GAT
from layers.attention import PoswiseFeedForwardNet, PositionalEncoding, MultiHeadAttention, get_attn_pad_mask


class Em_graph(nn.Module):
    '''
    enc_inputs：[batch_size,N_nodes,N_words]
    '''

    def __init__(self, opt, src_vocab_size, padding_idx, weight, max_words):
        super(Em_graph, self).__init__()
        self.embedding_dim = opt.embedding_dim
        self.padding_idx = padding_idx
        self.src_emb = nn.Embedding(src_vocab_size, opt.embedding_dim, padding_idx)
        self.pos_emb = PositionalEncoding(opt.embedding_dim, opt.dropout)
        self.self_attn = MultiHeadAttention(self.embedding_dim, opt.heads)
        self.linear = nn.Linear(self.embedding_dim * max_words, self.embedding_dim, bias=False)

    def reset_parameters(self):
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=self.embedding_dim ** -0.5)

    def forward(self, enc_inputs):
        enc_emb = self.src_emb(enc_inputs)
        enc_outputs = []
        for i, batch in enumerate(enc_emb):
            temp = self.pos_emb(batch)  
            temp_mask = get_attn_pad_mask(enc_inputs[i], enc_inputs[i],
                                          mask_pad=self.padding_idx)
            temp, _ = self.self_attn(temp, temp, temp, temp_mask)
            enc_outputs.append(temp.reshape(temp.size(0), -1))  
            enc_outputs = torch.stack(enc_outputs, dim=0)  
        enc_outputs = self.linear(enc_outputs)
        return enc_outputs


class GraphEncoderLayer(nn.Module):
    def __init__(self, in_features, ff_size, out_features, n_heads, dropout=0.1, alpha=0.01):
        super(GraphEncoderLayer, self).__init__()
        self.enc_graph_attn = GAT(in_features, ff_size, out_features, n_heads, dropout, alpha)
        self.pos_ffn = PoswiseFeedForwardNet(out_features, ff_size)

    def forward(self, enc_inputs, adjs):
        '''
        enc_inputs: [batch_size,N_nodes,in_features]=[batch_size, src_len, d_model]
        enc_outputs = [batch_size,N_nodes,out_features]
        adj: [N_nodes,N_nodes]
        '''
        enc_outputs = self.enc_graph_attn(enc_inputs, adjs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class GraphEncoder(nn.Module):
    def __init__(self, opt, src_vocab_size, padding_idx, weight, max_words):
        super(GraphEncoder, self).__init__()
        num_layers = 1  # opt.layers
        self.graph_emb = Em_graph(opt, src_vocab_size, padding_idx, weight, max_words)
        self.layers = nn.ModuleList(
            [GraphEncoderLayer(opt.embedding_dim, opt.ff_size, opt.embedding_dim, opt.heads, opt.dropout, opt.alpha) for
             _ in range(num_layers)])

    def forward(self, enc_inputs, adjs):
        '''
        enc_inputs: [batch_size,N_nodes,N_words]
        adjs:[batch_size,N_nodes,N_nodes]
        enc_outputs:[batch_size,N_nodes,out_features] = [batch_size, src_len, d_model]
        '''
        enc_outputs = self.graph_emb(enc_inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, adjs)
        return enc_outputs
