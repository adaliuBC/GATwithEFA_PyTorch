import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class GraphAttentionLayer_EFA(nn.Module):
    """
    GAT + EFA layer
    """

    def __init__(self, in_features, in_edge_features, out_features, dropout, alpha, lastact=False):
        #flowchart GAT
        super(GraphAttentionLayer_EFA, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.lastact = lastact
        self.bn = torch.nn.BatchNorm1d(out_features)

        self.Wh = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.Wh1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.Wh2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.Wf = nn.Parameter(torch.zeros(size=(in_edge_features, out_features)))
        self.ah = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.af = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.bf = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.Wh.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wh1.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wh2.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wf.data, gain=1.414)
        nn.init.xavier_uniform_(self.ah.data, gain=1.414)
        nn.init.xavier_uniform_(self.af.data, gain=1.414)
        nn.init.xavier_uniform_(self.bf.data, gain=1.414)

        self.activation = nn.LeakyReLU(self.alpha)

    def forward(self, input, edge_feat, adj):
        #compute h = input * W_h
        h = torch.mm(input, self.Wh)  #input: num*in_size, W:in_size*out_size, h:num*out_size
        N = h.size()[0]  #=stroke_num
        
        #compute cij
        h1 = torch.mm(input, self.Wh1)
        h2 = torch.mm(input, self.Wh2)
        ah_input = h1.repeat(1, N).view(N * N, -1) + h2.repeat(N, 1)      #W_h*h_i + W_h*H_j
        ah_input = ah_input.view(N, -1, self.out_features)              #N*N*32
        c = self.activation(torch.matmul(ah_input, self.ah).squeeze(2))  #N*N*32 · 32*1 = N*N*1-->N*N
        
        #compute c'ij
        input_edge = edge_feat.unsqueeze(2)            #N*N*1*19
        f = torch.matmul(input_edge, self.Wf)               #N*N*1*32           #=W_f·f_ij
        f = f + self.bf                                #N*N*1*32+N*N*1*32  #=W_f·f_ij + b_f
        af_input = self.activation(f)                   #N*N*1*32           #=δ(W_f·f_ij + b_f)
        cp = torch.matmul(af_input, self.af).squeeze(3)     #N*N*1              #=a_f·δ(W_f·f_ij + b_f)
        cp = self.activation(cp.squeeze(2))             #N*N                #=δ(a_f·δ(W_f·f_ij + b_f)))
        
        #compute cij & c'ij
        c = c + cp
        
        #compute output = h * attention adj matrix
        zero_vec = -9e15*torch.ones_like(c)      #ones_like：返回大小与input相同的1张量
        attention = torch.where(adj>0, c, zero_vec)  
        attention = F.softmax(attention, dim=1)  #α_ij
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #原有dropout
        h_prime = torch.matmul(attention, h)     #=∑ α_ij · Wh · h_j 
        
        h_prime = self.bn(h_prime)
        if self.lastact == True:
            return self.activation(h_prime)  #=δ(∑ α_ij·Wh·h_j)
        else:
            return h_prime  #=∑ α_ij·Wh·h_j
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    
class OutputLayer(nn.Module):  #GAT+EFA last layer
    def __init__(self, in_features, out_features):
        super(OutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.activation = F.log_softmax
    
    def forward(self, input):
        output = torch.mm(input, self.W)  #num*in · in*out --> num*out
        output = self.activation(output, dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, in_edge_features, out_features, dropout, alpha, lastact=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.lastact = lastact
        self.bn = torch.nn.BatchNorm1d(out_features)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        #origin GAT

    def forward(self, input, adj):
        
        h = torch.mm(input, self.W)  #input: num*in_size, W:in_size*out_size, h:num*out_size
        N = h.size()[0]  #=num

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)  #ones_like：返回大小与input相同的1张量
        attention = torch.where(adj>0, e, zero_vec)  
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        h_prime = self.bn(h_prime)
        if self.lastact:
            return F.elu(h_prime)
        else:
            return h_prime
        #origin GAT
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvLayer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        #init

    def forward(self, input, adj):
        mul = torch.mm(input, self.W)  #num*in · in*out --> num*out
        output = torch.mm(adj, mul) + self.b
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MLPLayer(nn.Module):
    """
    Simple MLP layer
    """

    def __init__(self, in_features, out_features):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        #init
        
        self.activation = F.relu

    def forward(self, input):
        mul = torch.mm(input, self.W)  #num*in · in*out --> num*out
        output = self.activation(mul + self.b)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
