# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""


import math
import torch
import torch.nn as nn
from model.ggnn import InOutGGNN
from model.InOutGat import InOutGATConv, InOutGATConv_intra
from torch_geometric.nn import GATConv, SGConv, GCNConv


class SRGNN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, dropout=0.5, negative_slope=0.2, heads=8, item_fusing=False):
        super(SRGNN, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.item_fusing = item_fusing
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # self.gated = InOutGGNN(self.hidden_size, num_layers=1)

        self.gcn = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.gcn2 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)

        self.gated = SGConv(in_channels=hidden_size, out_channels=hidden_size, K=2)
        # self.gated = InOutGATConv_intra(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout,
        #                           negative_slope=negative_slope, heads=heads, concat=True)
        # self.gated2 = InOutGATConv(in_channels=hidden_size * heads, out_channels=hidden_size, dropout=dropout,
        #                            negative_slope=negative_slope, heads=heads, concat=True, middle_layer=True)
        # self.gated3 = InOutGATConv(in_channels=hidden_size * heads, out_channels=hidden_size, dropout=dropout,
        #                            negative_slope=negative_slope, heads=heads, concat=False)

        self.W_1 = nn.Linear(self.hidden_size * 8, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size * 8, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(16 * self.hidden_size, self.hidden_size)

        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def rebuilt_sess(self, session_embedding, batchs, sess_item_index, seq_lens):
        sections = torch.bincount(batchs)
        split_embs = torch.split(session_embedding, tuple(sections.cpu().numpy()))
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))

        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)


    def get_h_s(self, hidden, seq_len):
        # split whole x back into graphs G_i
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(6)
        # print("v_n_repeat", v_n_repeat.size())
        # print("hidden", hidden.size())
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))    # |V|_i * 1

        s_g_whole = alpha * hidden    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        # print("torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)", torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1).size())
        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s

    def forward(self, data, hidden):
        edge_index, batch, edge_count, in_degree_inv, out_degree_inv, num_count, sess_item_index, seq_len = \
            data.edge_index, data.batch, data.edge_count, data.in_degree_inv, data.out_degree_inv,\
            data.num_count, data.sess_item_idx, data.sequence_len

        hidden = self.gated.forward(hidden, edge_index)
        # hidden = self.gcn.forward(hidden, edge_index)
        # hidden = self.gcn2.forward(hidden, edge_index)
        sess_embs = self.rebuilt_sess(hidden, batch, sess_item_index, seq_len)
        if self.item_fusing:
            return sess_embs
        else:
            return self.get_h_s(sess_embs, seq_len)
