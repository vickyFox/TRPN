from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(tt.arg.device))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).to(tt.arg.device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('device:', input.device, self.weight.device)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def norm(self, adj, symmetric=True):
        # A = A+I
        new_adj = adj + torch.eye(adj.size(0)).to(tt.arg.device)
        # 所有节点的度
        degree = new_adj.sum(1)
        if symmetric:
            # degree = degree^-1/2
            degree = torch.diag(torch.pow(degree, -0.5))
            return degree.mm(new_adj).mm(degree)
        else:
            # degree=degree^-1
            degree = torch.diag(torch.pow(degree, -1))
            return degree.mm(new_adj)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TRPN(nn.Module):
    def __init__(self, n_feat, n_queries, hidden_layers = [640,320,320,160]):
        super(TRPN, self).__init__()
        #self.layer_last = nn.Sequential(nn.Linear(in_features=512,
        #                                out_features=128, bias=True),
        #                                nn.BatchNorm1d(128))

        self.fc_1 = nn.Sequential(nn.Linear(in_features=n_feat * 2, out_features=hidden_layers[0], bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1], bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=hidden_layers[1], out_features=1, bias=True),
                                nn.Sigmoid())

        self.fc_2 = nn.Sequential(nn.Linear(in_features=n_feat * (n_queries + 1), out_features=hidden_layers[2], bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=hidden_layers[2], out_features=hidden_layers[3], bias=True),
                                nn.ReLU(),
                                nn.Linear(in_features=hidden_layers[3], out_features=n_queries, bias=True),
                                nn.Sigmoid())

        self.gc = GraphConvolution(n_feat * (n_queries + 1), n_feat * (n_queries + 1))

    def forward(self, node_feat, adj):
        # node_feat: batch_size(num_tasks) x num_samples x in_features
        # adj: batch_size(num_tasks) x num_supports x num_supports  [0, 1]
        #node_feat = self.layer_last(node_feat.view(-1,512)).view(-1, 30, 128)
        num_tasks = node_feat.size(0)
        num_samples = node_feat.size(1)
        num_supports = adj.size(1)
        num_queries = num_samples - num_supports
        in_features_2 = node_feat.size(2) * 2

        x_i = node_feat.unsqueeze(2).repeat(1, 1, node_feat.size(1), 1) 
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.cat((x_i, x_j), -1)

        # batch_size x num_samples x (in_features * (num_queries + 1))
        gcn_input_feat = node_feat
        for i in range(num_queries):
            gcn_input_feat = torch.cat((gcn_input_feat, node_feat[:, num_supports + i, :].unsqueeze(1).repeat(1, num_samples, 1)), -1)

        learned_score_list = []
        query_score_list = []
        for i in range(num_tasks):
            # num_samples x num_samples
            learned_score = self.fc_1(x_ij[i].contiguous().view(num_samples ** 2, in_features_2)).view(num_samples, num_samples)
            learned_adj = learned_score.clone()
            ones = torch.ones(learned_adj[:num_supports, :num_supports].size()).to(tt.arg.device)
            if tt.arg.num_unlabeled >0:
                learned_adj[:num_supports, :num_supports] = torch.where(adj[i] == 1.0, ones ,learned_adj[:num_supports, :num_supports])
                learned_adj[:num_supports, :num_supports] = torch.where(adj[i] == 0,-learned_adj[:num_supports, :num_supports],learned_adj[:num_supports, :num_supports])
            else:
                learned_adj[:num_supports, :num_supports] = torch.where(adj[i] > 0, ones, -learned_adj[:num_supports, :num_supports])
            # num_samples x num_queries
            query_score = self.fc_2(F.relu(self.gc(gcn_input_feat[i], learned_adj)))
            learned_score_list.append(learned_score)
            query_score_list.append(query_score)

        # query_score_list: batch_size x num_queries x num_samples
        # learned_score_list: batch_size x num_samples x num_samples
        return torch.stack(query_score_list, 0).transpose(1, 2), torch.stack(learned_score_list, 0)


