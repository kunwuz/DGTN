from model.multi_sess import GroupGraph
from model.srgnn import SRGNN

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size, n_node, using_represent, item_fusing=False):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.n_node = n_node
        self.using_represent = using_represent
        self.item_fusing = item_fusing


        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, h_s, h_group, final_s, item_embedding_table):
        emb = item_embedding_table.weight.transpose(1, 0)
        if self.item_fusing:
            z_i_hat = torch.mm(final_s, emb)
        else:
            gate = F.sigmoid(self.W_2(h_s) + self.W_3(h_group))
            sess_rep = h_s * gate + h_group * (1 - gate)
            if self.using_represent == 'comb':
                z_i_hat = torch.mm(sess_rep, emb)
            elif self.using_represent == 'h_s':
                z_i_hat = torch.mm(h_s, emb)
            elif self.using_represent == 'h_group':
                z_i_hat = torch.mm(h_group, emb)
            else:
                print("invalid represent type")
                exit()


        return z_i_hat,


class ItemFusing(nn.Module):
    def __init__(self, hidden_size):
        super(ItemFusing, self).__init__()
        self.hidden_size = hidden_size
        self.use_rnn = False
        self.Wf1 = nn.Linear(self.hidden_size * 8, self.hidden_size)
        self.Wf2 = nn.Linear(self.hidden_size * 8, self.hidden_size)

        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.rnn = torch.nn.GRUCell(hidden_size, hidden_size, bias=True)

    def forward(self, inter_item_emb, intra_item_emb, seq_len):
        final_emb = self.item_fusing(inter_item_emb, intra_item_emb)
        final_s = self.get_final_s(final_emb, seq_len)
        return final_s

    def item_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        if self.use_rnn:
            final_emb = self.rnn(local_emb, global_emb)
        else:
            gate = F.sigmoid(self.Wf1(local_emb) + self.Wf2(global_emb))
            # final_emb = local_emb * gate + global_emb * (1 - gate)
            final_emb = self.Wf1(local_emb) * gate + self.Wf2(global_emb) * (1 - gate)

        return final_emb

    def get_final_s(self, hidden, seq_len):
        hidden = torch.split(hidden, tuple(seq_len.cpu().numpy()))
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))    # |V|_i * 1
        s_g_whole = alpha * hidden    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s


class GraphModel(nn.Module):
    def __init__(self, opt, n_node):
        super(GraphModel, self).__init__()
        self.hidden_size, self.n_node = opt.hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.dropout = opt.gat_dropout
        self.negative_slope = opt.negative_slope
        self.heads = opt.heads
        self.item_fusing = opt.item_fusing

        self.srgnn = SRGNN(self.hidden_size, n_node=n_node, item_fusing=opt.item_fusing)
        self.group_graph = GroupGraph(self.hidden_size, dropout=self.dropout, negative_slope=self.negative_slope,
                                      heads=self.heads, item_fusing=opt.item_fusing)
        self.fuse_model = ItemFusing(self.hidden_size)
        self.e2s = Embedding2Score(self.hidden_size, n_node, opt.using_represent, opt.item_fusing)

        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        if self.item_fusing:
            x = data.x - 1
            embedding = self.embedding(x)
            embedding = embedding.squeeze()
            intra_item_emb = self.srgnn(data, embedding)

            mt_x = data.mt_x - 1

            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()

            inter_item_emb = self.group_graph.forward(embedding, data)

            final_s = self.fuse_model.forward(inter_item_emb, intra_item_emb, data.sequence_len)

            scores = self.e2s(h_s=None, h_group=None, final_s=final_s, item_embedding_table=self.embedding)

        else:
            x = data.x - 1
            embedding = self.embedding(x)
            embedding = embedding.squeeze()
            h_s = self.srgnn(data, embedding)

            mt_x = data.mt_x - 1

            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()

            h_group = self.group_graph.forward(embedding, data)
            scores = self.e2s(h_s=h_s, h_group=h_group, final_s=None, item_embedding_table=self.embedding)

        return scores[0]

