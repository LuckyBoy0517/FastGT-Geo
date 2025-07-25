import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
from dgl.nn.pytorch import GraphConv,NNConv
import os
import dgl
import random as rd

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index=data.graph['edge_index']
        edge_weight=data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x=conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x

class MPNN(nn.Module):
    def __init__(self, aggregator_type,node_in_feats, node_hidden_dim, edge_input_dim, edge_hidden_dim,num_step_message_passing,gconv_dp,edge_dp,nn_dp1):
        #原GCN是对基础的GraphConv(in_feats, hidden_size)的包装，多了一个linear_size参数，代表最后卷积层输出后，额外加一层的
        # FC，好得到最终的表示，非常简单。。。
        super(MPNN, self).__init__()
        self.lin0 = nn.Linear(node_in_feats, node_hidden_dim)#65,32
        self.num_step_message_passing=num_step_message_passing#层数 开始测试1层即可
        edge_network = nn.Sequential(
            # edge的处理网络 一般是MLP，输入自然是edge_input_dim，
            # 最后的输出必须是图卷积层的节点输入特征x节点隐藏特征
            # 原文注释 注意in_feats和out_feats就是NNConv(in_feats=node_in_feats,out_feats=node_hidden_dim,...)
            # edge_func : callable activation function/layer
            # Maps each edge feature to a vector of shape
            # ``(in_feats * out_feats)`` as weight to compute messages.
            #我们这里在原始的节点输入dim基础上，通过一层线性层将其转换为指定的维度作为图卷积层的输入node_hidden_dim
            #参考qm9_nn 多层，edgefunc也是一样的 我的想法还没搞清楚；minist则是不一样的；mpnndgl也是一样的

            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=edge_dp),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim),
            nn.Dropout(p=edge_dp))#1-4-32x32

        self.conv = NNConv(in_feats=node_hidden_dim,#32
               out_feats=node_hidden_dim,#32
               edge_func=edge_network,#32x32
               aggregator_type=aggregator_type)

        # lat,lon分别用一层mlp算一下即可

        self.y_linear = nn.Linear(node_hidden_dim, 2)#4-4
        self.bn = nn.BatchNorm1d(node_hidden_dim)
        # self.y1_predict = nn.Linear(linear_size, 1)

        # # 分别用一层mlp算一下即可
        # self.y2_linear = nn.Linear(node_hidden_dim, linear_size)
        # self.y2_predict = nn.Linear(linear_size, 1)


        self.gnn_dropout = nn.Dropout(p=gconv_dp)#dropout
        self.nn_dropout = nn.Dropout(p=nn_dp1)
        # self.nn_dropout2 = nn.Dropout(p=nn_dp2)

    def forward(self, g, n_feat, e_feat):
        out = torch.relu(self.lin0(n_feat))  # (B1, H1)

        for i in range(self.num_step_message_passing):
            out = torch.relu(self.conv(g, out, e_feat))
            out = self.gnn_dropout(out)# (B1, H1)

        y_bn = self.bn(out)
        #y = out
        #y_si = torch.sigmoid(self.y_linear(y_bn))
        return y_bn
        #return y
        #**********y_sigmoid = torch.sigmoid(self.y_linear(y_bn))#我自己修改的


        # y_bn = self.bn(self.y_linear(out))
        # y_sigmoid = torch.sigmoid(y_bn)

        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (2878x32 and 4x4)
        # y2_relu = torch.sigmoid(self.y2_linear(out))
        # y2_relu_dp = self.nn_dropout2(y2_relu)
        # y1_predict = self.y1_predict(y1_relu_dp)
        # y2_predict = self.y2_predict(y2_relu_dp)

        #*****return y_sigmoid