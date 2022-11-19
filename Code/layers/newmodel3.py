'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> newmodel3
@Author ：MollyShuu
@Date   ：2022/11/1 14:56
@IDE    ：PyCharm
==================================================
'''
import torch
import torch.nn as nn
from typing import Dict
from layers.graph_encoder import GraphEncoder
from layers.cross_trans_previous_decoder import GraphDecoder
from layers.encoder import Encoder


class GraphTransformer(nn.Module):
    def __init__(self, opt, src_vocab_size, tgt_vocab_size, padding_idx, src_weight, tgt_weight):
        super(GraphTransformer, self).__init__()
        self.graph_encoder = GraphEncoder(opt, src_vocab_size, padding_idx['src'],
                                          weight=src_weight, max_words=opt.max_swords)
        self.cpt_encoder = Encoder(opt, src_vocab_size, padding_idx['src'],
                                   weight=src_weight)
        self.decoder = GraphDecoder(opt, tgt_vocab_size, padding_idx['tgt'],
                                    weight=tgt_weight)
        self.projection = nn.Linear(opt.embedding_dim, tgt_vocab_size, bias=False)
        self.sm = nn.Softmax(dim=-1)

        self.tgt_vocab_size = tgt_vocab_size
        self.opt = opt


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, enc_inputs, en_adjs, cpt_inputs, dec_inputs, probs, idxes):
        '''
        enc_inputs: [batch_size, N_nodes,N_words]
        cpt_inputs:[batch_size,src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        dec_inputs = dec_inputs[:, : -1]
        enc_outputs = self.graph_encoder(enc_inputs, en_adjs)
        cpt_outputs, enc_cpt_attns = self.cpt_encoder(cpt_inputs)
        dec_outputs, p_range, result2, dec_enc_attns, weight = self.decoder(dec_inputs, enc_inputs,
                                                                            enc_outputs, cpt_inputs,
                                                                            cpt_outputs)
 
        dec_logits = self.sm(self.projection(dec_outputs))  


        bsize, src_len, _ = cpt_outputs.size()
        tmp_trans_scores = cpt_outputs[:, :, 0].unsqueeze(2).expand(bsize, src_len, self.tgt_vocab_size) * 0.0
        tmp_trans_scores.scatter_add_(2, idxes, probs)  
        weight = weight.float()  

        trans_scores = torch.matmul(weight, tmp_trans_scores) 
        del tmp_trans_scores, weight, dec_enc_attns, dec_outputs

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        final_scores = torch.log(
            p_range * trans_scores + (1 - p_range) * dec_logits)  

 
    @classmethod
    def load_model(cls, opt, pad_ids: Dict[str, int], vocab_sizes: Dict[str, int], weights, device,
                   checkpoint=None):
        model = cls(opt, vocab_sizes['src'], vocab_sizes['tgt'], pad_ids, weights['src_weight'], weights['tgt_weight'])
        # 多GPU并行计算请启用以下代码
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model, device_ids=[
        #         0])  
        model.to(device)
        if opt.train_from and checkpoint is None:
            checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"], False)
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
