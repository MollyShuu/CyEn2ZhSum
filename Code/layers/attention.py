import torch
import math
import numpy as np
import torch.nn as nn
from Utils import get_device

device = get_device()


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=210):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, timestep=0):
        x = x * math.sqrt(self.embedding_dim) + self.pe[timestep:timestep + x.size(1)]
        return x


def get_attn_pad_mask(seq_q, seq_k, mask_pad=0):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(mask_pad).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k)  



def get_graph_attn_pad_mask(seq_q, seq_k, mask_pad=0):

    batch_size, len_q = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    mask = torch.sum(seq_k.data.eq(mask_pad), dim=-1).unsqueeze(1)
    pad_graph_attn_mask = mask.data.eq(len_k)  

    return pad_graph_attn_mask.expand(batch_size, len_q, len_k)  



def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  
    subsequence_mask = torch.from_numpy(subsequence_mask).byte().to(device)
    return subsequence_mask  



class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_per_head):
        super(ScaledDotProductAttention, self).__init__()
        self.dim_per_head = dim_per_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.dim_per_head)  
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(scores)
        context = torch.matmul(attn, V)  
        return context, attn


class PoswiseFeedForwardNet(nn.Module):  
    def __init__(self, embedding_dim, ff_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.linear_in = nn.Linear(embedding_dim, ff_size, bias=True)
        self.linear_out = nn.Linear(ff_size, embedding_dim, bias=True)
        self.relu = nn.ReLU()  
        self.layernorm = nn.LayerNorm(embedding_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, inputs):
        residual = inputs
        output = self.linear_in(inputs)
        output = self.relu(output)
        output = self.linear_out(output)
        return self.layernorm(output + residual) 

class MultiHeadAttention(nn.Module):  
    def __init__(self, embed_dim, n_heads, dropout=0.1, bias: bool = True):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = embed_dim // n_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        # self.reset_parameters()
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-05, elementwise_affine=True)

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.W_Q.weight)
    #     nn.init.xavier_uniform_(self.W_K.weight)
    #     nn.init.xavier_uniform_(self.W_V.weight)
    #     nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.k_proj(input_Q).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(
            1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.v_proj(input_K).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(
            1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.q_proj(input_V).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(
            1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1,
            1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]


        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.dim_per_head)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = self.dropout(self.softmax(scores))
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]

        context = context.transpose(1, 2).reshape(
            batch_size, -1,
            self.n_heads * self.dim_per_head)  # context: [batch_size, len_q, n_heads * d_v]

        output = self.out_proj(context)  # [batch_size, len_q, d_model]
        avg_weights = torch.sum(attn, dim=1) / self.n_heads
        attn = avg_weights
        return self.layernorm(output + residual), attn
