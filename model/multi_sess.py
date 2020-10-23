import torch
from torch import nn
from torch.nn import Module, Parameter
from model.ggnn import InOutGGNN
from model.InOutGat import InOutGATConv
from torch_geometric.nn.conv import GATConv, GatedGraphConv, GCNConv, SGConv


class GroupGraph(Module):
    def __init__(self, hidden_size, dropout=0.5, negative_slope=0.2, heads=8, item_fusing=False):
        super(GroupGraph, self).__init__()
        self.hidden_size = hidden_size
        self.item_fusing = item_fusing

        self.W_1 = nn.Linear(8 * self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(8 * self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(16 * self.hidden_size, self.hidden_size)

        # self.gat = GATConv(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=True)
        # self.gat2 = GATConv(in_channels=hidden_size*heads, out_channels=hidden_size*heads, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=False)
        # self.gat3 = GATConv(in_channels=hidden_size*heads, out_channels=hidden_size, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=True)
        # self.gat_out = GATConv(in_channels=hidden_size*heads, out_channels=hidden_size, dropout=dropout, negative_slope=negative_slope, heads=heads, concat=False)
        # self.gated = InOutGGNN(self.hidden_size, num_layers=2)
        self.gcn = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.gcn2 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)

        self.sgcn = SGConv(in_channels=hidden_size, out_channels=hidden_size, K=2)
        # self.gat = InOutGATConv(in_channels=hidden_size, out_channels=hidden_size, dropout=dropout,
        #                           negative_slope=negative_slope, heads=heads, concat=True)
        # self.gat2 = InOutGATConv(in_channels=hidden_size * heads, out_channels=hidden_size, dropout=dropout,
        #                            negative_slope=negative_slope, heads=heads, concat=False)
        #

    def group_att_old(self, session_embedding, node_num, batch_h_s):  # hs: # batch_size x latent_size
        v_i = torch.split(session_embedding, tuple(node_num))    # split whole x back into graphs G_i
        h_s_repeat = tuple(h_s.view(1, -1).repeat(nodes.shape[0], 1) for h_s, nodes in zip(batch_h_s, v_i))    # repeat |V|_i times for the last node embedding

        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(h_s_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(node_num.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        return torch.cat(s_g, dim=0)

    def group_att(self, session_embedding, hidden, node_num, num_count):  # hs: # batch_size x latent_size
        v_i = torch.split(session_embedding, tuple(node_num))    # split whole x back into graphs G_i
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(sess_nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for sess_nodes, nodes in zip(hidden, v_i))    # repeat |V|_i times for the last node embedding

        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = num_count.view(-1, 1) * alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(node_num.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))

        return h_s


    def rebuilt_sess(self, session_embedding, node_num, sess_item_index, seq_lens):
        split_embs = torch.split(session_embedding, tuple(node_num))
        sess_item_index = torch.split(sess_item_index, tuple(seq_lens.cpu().numpy()))

        rebuilt_sess = []
        for embs, index in zip(split_embs, sess_item_index):
            sess = tuple(embs[i].view(1, -1) for i in index)
            sess = torch.cat(sess, dim=0)
            rebuilt_sess.append(sess)
        return tuple(rebuilt_sess)

    def get_h_group(self, hidden, seq_len):
        # split whole x back into graphs G_i
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(5)
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))    # |V|_i * 1
        s_g_whole = alpha * hidden    # |V|_i * hidden_size
        # s_g_whole = hidden
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        # s_g = tuple(torch.mean(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        h_s = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # h_s = torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)
        return h_s

    def h_mean(self, hidden, node_num):
        split_embs = torch.split(hidden, tuple(node_num))
        means = []
        for embs in split_embs:
            mean = torch.mean(embs, dim=0)
            means.append(mean)

        means = torch.cat(tuple(means), dim=0).view(len(split_embs), -1)

        return means

    def forward(self, hidden, data):
        # edge_index, node_num, batch, sess_item_index, seq_lens, sess_masks = \
        #     data.mt_edge_index, data.mt_node_num, data.batch, data.mt_sess_item_idx, data.sequence_len, data.sess_masks
        edge_index, node_num, batch, sess_item_index, seq_lens = \
            data.mt_edge_index, data.mt_node_num, data.batch, data.mt_sess_item_idx, data.sequence_len


        # edge_count, in_degree_inv, out_degree_inv = data.mt_edge_count, data.mt_in_degree_inv, data.mt_out_degree_inv
        # hidden = self.gat.forward(hidden, edge_index, sess_masks)
        # hidden = self.gat2.forward(hidden, edge_index)
        # hidden = self.gat3.forward(hidden, edge_index)

        # hidden = self.gat.forward(hidden, edge_index, sess_masks)

        hidden - self.sgcn(hidden, edge_index)
        # hidden = self.gcn.forward(hidden, edge_index)
        # hidden = self.gcn2.forward(hidden, edge_index)

        # hidden = self.gat.forward(hidden, edge_index)
        # hidden = self.gated.forward(hidden, edge_index, [edge_count * in_degree_inv, edge_count * out_degree_inv])
        # hidden = self.gated.forward(hidden, edge_index)

        # hidden = self.gat1.forward(hidden, edge_index)

        sess_hidden = self.rebuilt_sess(hidden, node_num, sess_item_index, seq_lens)

        if self.item_fusing:
            return sess_hidden
        else:
            return self.get_h_group(sess_hidden, seq_lens)


