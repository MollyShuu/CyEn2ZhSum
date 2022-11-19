import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """
    input: (B,N,in_features)
    output: (B,N,out_features)
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.alpha = alpha
        self.concat = concat  


        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  


        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)  
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  
        adj:   [B,N, N] 
        """

        h = torch.matmul(inp, self.W)  
        N = h.size()[1]  
        del inp
        #if hasattr(torch.cuda, 'empty_cache'):
          #  torch.cuda.empty_cache()
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)],
                            dim=2).view(-1, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))


        zero_vec = -1e12 * torch.ones_like(e)  

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        del a_input, e, zero_vec
 
        attention = self.softmax(attention) 
        attention = self.dropout(attention)  
        h_prime = torch.matmul(attention, h)  
        del attention, h
        if self.concat:
            return self.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads, dropout=0.1, alpha=0.01):
        """
        Input：[batch_size,N_nodes,in_features]
        Output：[batch_size,N_nodes,out_features]
        """
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        n_heads = 3  


        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  
        x = self.dropout(x)  
        x = self.out_att(x, adj)
        x = self.elu(x)  
        return x
