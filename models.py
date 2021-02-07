import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from layers import GraphAttentionLayer_EFA, GraphAttentionLayer, OutputLayer, GraphConvLayer, MLPLayer


class GAT_EFA(nn.Module):
    def __init__(self, nfeat, nedgef, nhid, nclass, dropout, alpha, nheads, noutheads, nlayer):
        """Dense version of GAT."""
        super(GAT_EFA, self).__init__()
        self.dropout = dropout
        self.nlayer = nlayer
        self.nheads = nheads
        self.noutheads = noutheads
        
        self.attentions = []
        #in layer
        self.attentions.append([GraphAttentionLayer_EFA(nfeat, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions[0]):
            self.add_module('attention_{}_{}'.format(0, i), attention)
        #attention layers
        for j in range(1, nlayer-1):
            self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
            for i, attention in enumerate(self.attentions[j]):
                self.add_module('attention_{}_{}'.format(j, i), attention)
        #last attention layer
        self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=False) for _ in range(noutheads)])
        for i, attention in enumerate(self.attentions[nlayer-1]):
            self.add_module('attention_{}_{}'.format(nlayer-1, i), attention)
        #output layer
        self.out_layer = OutputLayer(nhid, nclass)
    
        #self.activation = nn.LeakyReLU(alpha)
        self.activation = F.relu

    def forward(self, x, edge_feats, adj):
        #input_layer
        x = torch.cat([att(x, edge_feats, adj) for att in self.attentions[0]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        #hidden layer
        for j in range(1, self.nlayer-1):
            mid = torch.cat([att(x, edge_feats, adj) for att in self.attentions[j]], dim=1)
            x = mid + x  #residual connections
            x = F.dropout(x, self.dropout, training=self.training)
        
        #last hidden layer
        x = torch.mean(torch.stack([att(x, edge_feats, adj) for att in self.attentions[self.nlayer-1]]), 0)
        x = self.activation(x)  #h_i=δ(avg(∑ α_ij·Wh·h_j))
        x = F.dropout(x, self.dropout, training=self.training)
        
        #output layer
        x = self.out_layer(x)
        #return F.log_softmax(x, dim=1)
        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nedgef, nhid, nclass, dropout, alpha, nheads, noutheads, nlayer):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayer = nlayer
        self.attentions = []
        self.attentions.append([GraphAttentionLayer(nfeat, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions[0]):
            self.add_module('attention_{}_{}'.format(0, i), attention)
        for j in range(1, nlayer):
            self.attentions.append([GraphAttentionLayer(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
            for i, attention in enumerate(self.attentions[j]):
                self.add_module('attention_{}_{}'.format(j, i), attention)

        self.out_att = GraphAttentionLayer(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=False)

    def forward(self, x, edge_feats, adj):
        x = torch.cat([att(x, adj) for att in self.attentions[0]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        for j in range(1, self.nlayer):
            mid = torch.cat([att(x, adj) for att in self.attentions[j]], dim=1)
            x = mid + x
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nlayer):
        """GCN."""
        super(GCN, self).__init__()
        self.dropout = dropout
        self.layers = []
        self.layers.append(GraphConvLayer(nfeat, nhid))
        self.add_module('gcn_{}'.format(0), self.layers[0])
        self.layers.append(GraphConvLayer(nhid, nclass))
        self.add_module('gcn_{}'.format(1), self.layers[1])
        
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_feats, adj):  #useless edge_feats
        #in_layer
        x = self.act(self.layers[0](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[1](x, adj)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha):
        """MLP net."""
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layers = []
        self.layers.append(MLPLayer(nfeat, nhid))
        self.add_module('MLPlayer_{}'.format(0), self.layers[0])
        self.layers.append(MLPLayer(nhid, nclass))
        self.add_module('MLPlayer_{}'.format(1), self.layers[1])
        
    def forward(self, x, edge_feats, adj):
        x = self.layers[0](x)  # = relu(W*x+b)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[1](x)
        return F.log_softmax(x, dim=1)
