'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> cross_trans_previous_decoder.py
@Author ：MollyShuu
@Date   ：2021/6/10 21:44
@IDE    ：PyCharm
==================================================
'''
import torch
import torch.nn as nn
from layers.attention import MultiHeadAttention, PoswiseFeedForwardNet, PositionalEncoding, get_graph_attn_pad_mask, \
    get_attn_pad_mask, get_attn_subsequence_mask

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, ff_size):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(embedding_dim, heads)
        # dec和sentence图的attention
        self.dec_enc_attn = MultiHeadAttention(embedding_dim, heads)
        # dec和concepts图的attention
        self.dec_cpt_attn = MultiHeadAttention(embedding_dim, heads)
        self.pos_ffn = PoswiseFeedForwardNet(embedding_dim, ff_size)

    def forward(self, dec_inputs, enc_outputs, cpt_outputs, dec_self_attn_mask, dec_enc_attn_mask, dec_cpt_attn_mask,
                previous=None):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        cpt_outputs:[batch_size,N_nodes,embed_dim=d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        all_input = dec_inputs if previous is None else torch.cat((previous, dec_inputs), dim=1)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, all_input, all_input, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs, dec_cpt_attn = self.dec_cpt_attn(dec_outputs, cpt_outputs, cpt_outputs, dec_cpt_attn_mask)
 
        dec_outputs = self.pos_ffn(dec_outputs)  
        return dec_outputs, all_input, dec_enc_attn, dec_cpt_attn


class GraphDecoder(nn.Module):
    def __init__(self, opt, tgt_vocab_size, padding_idx, weight):
        super(GraphDecoder, self).__init__()
        self.heads = opt.heads
        self.embedding_dim = opt.embedding_dim
        if (opt.emd_type == 1):
            self.tgt_emb = nn.Embedding(tgt_vocab_size, opt.embedding_dim, padding_idx)
          elif (opt.emd_type == 2):
            self.tgt_emb = nn.Embedding(tgt_vocab_size, opt.embedding_dim, padding_idx)
            self.tgt_emb.weight.data.copy_(weight)
            self.tgt_emb.weight.requires_grad = True
        self.pos_emb = PositionalEncoding(opt.embedding_dim, opt.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(opt.embedding_dim, opt.heads, opt.ff_size) for _ in range(opt.layers)])

        self.range_linear1 = nn.Linear(opt.embedding_dim, opt.embedding_dim, bias=False)
        self.range_linear2 = nn.Linear(opt.embedding_dim, 1, bias=False)
        self.range_gate = nn.Sigmoid()
        self.padding_idx = padding_idx

    def reset_parameters(self):
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=self.embedding_dim ** -0.5)

    def forward(self, dec_inputs, enc_inputs, enc_outputs, cpt_inputs, cpt_outputs, previous=None, timestep=0):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, N_nodes,N_words]
        cpt_intpus: [batch_size, N_nodes,N_words]
        enc_outputs: [batsh_size, src_len, d_model] =[batch_size,N_nodes,out_features]
        dec_outputs: [batch_size, tgt_len, d_model] = [batch_size,N_nodes,out_features]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  
        dec_outputs = self.pos_emb(dec_outputs, timestep)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs,
                                                   mask_pad=self.padding_idx) 
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs) 
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  

        dec_enc_attn_mask = get_graph_attn_pad_mask(dec_inputs, enc_inputs,
                                                    mask_pad=self.padding_idx) 
        dec_cpt_attn_mask = get_attn_pad_mask(dec_inputs, cpt_inputs, mask_pad=self.padding_idx)
        saved_inputs = []
        for i, layer in enumerate(self.layers):
             prev_layer = None if previous is None else previous[:, i]
            dec_self_attn_mask = dec_self_attn_mask if previous is None else torch.zeros(1).byte().cuda()
            dec_outputs, all_input, dec_enc_attn, dec_cpt_attn = layer(dec_outputs, enc_outputs, cpt_outputs,
                                                                       dec_self_attn_mask, dec_enc_attn_mask,
                                                                       dec_cpt_attn_mask, prev_layer)
            saved_inputs.append(all_input)
        result2 = torch.stack(saved_inputs, dim=1)
        del saved_inputs
        p_range = self.range_linear2(self.range_linear1(dec_outputs))
        p_range = self.range_gate(p_range) 
        return dec_outputs, p_range, result2, dec_enc_attn, dec_cpt_attn
