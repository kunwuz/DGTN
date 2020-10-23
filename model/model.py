from model.multi_sess import GroupGraph
from model.srgnn import SRGNN

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size, n_node, using_represent, item_fusing):
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
                raise NotImplementedError

        return z_i_hat,


class ItemFusing(nn.Module):
    def __init__(self, hidden_size):
        super(ItemFusing, self).__init__()
        self.hidden_size = hidden_size
        self.use_rnn = True
        self.Wf1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wf2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.rnn = torch.nn.GRUCell(hidden_size, hidden_size, bias=True)

    def forward(self, intra_item_emb, inter_item_emb, seq_len):
        final_emb = self.item_fusing(intra_item_emb, inter_item_emb)
        # final_emb = self.avg_fusing(intra_item_emb, inter_item_emb)
        final_s = self.get_final_s(final_emb, seq_len)
        return final_s

    def item_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        if self.use_rnn:
            final_emb = self.rnn(local_emb, global_emb)
        else:
            gate = F.sigmoid(self.Wf1(local_emb) + self.Wf2(global_emb))
            final_emb = local_emb * gate + global_emb * (1 - gate)

        return final_emb

    def cnn_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = embedding.permute(0, 2, 1)
        embedding = self.conv(embedding).permute(0, 2, 1)
        embedding = self.W_c(embedding).squeeze()
        return embedding

    def max_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = torch.max(embedding, dim=2)[0]
        return embedding

    def avg_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = (local_emb + global_emb) / 2
        return embedding

    def concat_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.cat([local_emb, global_emb], dim=1)
        embedding = self.W_4(embedding)
        return embedding
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


class NARM(nn.Module):
    def __init__(self, opt):
        super(NARM, self).__init__()
        self.hidden_size = opt.hidden_size
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

    def sess_att(self, hidden, ht, mask):
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        hs = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # hs = torch.sum(alpha * hidden, 1)
        return hs

    def padding(self, intra_item_embs, inter_item_embs, seq_lens):
        inter_padded, intra_padded = [], []
        max_len = max(seq_lens).detach().cpu().numpy()
        for intra_item_emb, inter_item_emb, seq_len in zip(intra_item_embs, inter_item_embs, seq_lens):
            if intra_item_emb.size(0) < max_len:
                pad_vec = torch.zeros(max_len - intra_item_emb.size(0), self.hidden_size)
                pad_vec = pad_vec.to('cuda')
                intra_item_emb = torch.cat((intra_item_emb, pad_vec), dim=0)
                inter_item_emb = torch.cat((inter_item_emb, pad_vec), dim=0)
            inter_padded.append(inter_item_emb.unsqueeze(dim=0))
            intra_padded.append(intra_item_emb.unsqueeze(dim=0))
        inter_padded = torch.cat(tuple(inter_padded), dim=0)
        intra_padded = torch.cat(tuple(intra_padded), dim=0)
        item_embs = torch.cat((inter_padded, intra_padded), dim=-1)
        return item_embs

    def get_h_s(self, padded, seq_lens, masks):
        outputs, _ = self.gru(padded)
        output_last = outputs[torch.arange(seq_lens.shape[0]).long(), seq_lens - 1]
        hs = self.sess_att(outputs, output_last, masks)
        return hs

    def forward(self, intra_item_embs, inter_item_embs, seq_lens):
        max_len = max(seq_lens).detach().cpu().numpy()
        masks = [[1] * le + [0] * (max_len - le) for le in seq_lens.detach().cpu().numpy()]
        masks = torch.tensor(masks).to('cuda')
        item_embs = self.padding(intra_item_embs, inter_item_embs, seq_lens)
        return self.get_h_s(item_embs, seq_lens, masks)


class CNNFusing(nn.Module):
    def __init__(self, hidden_size, num_filters):
        super(CNNFusing, self).__init__()
        self.hidden_size = hidden_size
        self.num_filters = num_filters

        self.Wf1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wf2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.W_4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # self.conv = torch.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(1, 2))
        self.conv = torch.nn.Conv1d(in_channels=2, out_channels=self.num_filters, kernel_size=1)
        self.W_c = nn.Linear(self.num_filters, 1)
    # def forward(self, inter_item_emb, intra_item_emb, seq_len):
    #     final_emb = self.cnn_fusing(inter_item_emb, intra_item_emb)
    #     final_s = self.get_final_s(final_emb, seq_len)
    #     return final_s
    def forward(self, intra_item_emb, inter_item_emb, seq_len):
        # final_emb = self.cnn_fusing(intra_item_emb, inter_item_emb)
        # final_emb = self.concat_fusing(intra_item_emb, inter_item_emb)
        # final_emb = self.avg_fusing(intra_item_emb, inter_item_emb)
        final_emb = self.max_fusing(intra_item_emb, inter_item_emb)
        # final_emb = intra_item_emb
        final_s = self.get_final_s(final_emb, seq_len)
        return final_s

    def cnn_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = embedding.permute(0, 2, 1)
        embedding = self.conv(embedding).permute(0, 2, 1)
        embedding = self.W_c(embedding).squeeze()
        return embedding

    def max_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.stack([local_emb, global_emb], dim=2)
        embedding = torch.max(embedding, dim=2)[0]
        return embedding

    def avg_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = (local_emb + global_emb) / 2
        return embedding

    def concat_fusing(self, local_emb, global_emb):
        local_emb = torch.cat(local_emb, dim=0)
        global_emb = torch.cat(global_emb, dim=0)
        embedding = torch.cat([local_emb, global_emb], dim=1)
        embedding = self.W_4(embedding)
        return embedding

    def get_final_s(self, hidden, seq_len):
        hidden = torch.split(hidden, tuple(seq_len.cpu().numpy()))
        v_n = tuple(nodes[-1].view(1, -1) for nodes in hidden)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in hidden)
        v_n_repeat = torch.cat(v_n_repeat, dim=0)
        hidden = torch.cat(hidden, dim=0)

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(v_n_repeat) + self.W_2(hidden)))  # |V|_i * 1
        s_g_whole = alpha * hidden  # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(seq_len.cpu().numpy()))  # split whole s_g into graphs G_i
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
        self.num_filters = opt.num_filters

        self.srgnn = SRGNN(self.hidden_size, n_node=n_node, item_fusing=opt.item_fusing)
        self.group_graph = GroupGraph(self.hidden_size, dropout=self.dropout, negative_slope=self.negative_slope,
                                      heads=self.heads, item_fusing=opt.item_fusing)
        self.fuse_model = ItemFusing(self.hidden_size)
        self.narm = NARM(opt)
        self.cnn_fusing = CNNFusing(self.hidden_size, self.num_filters)
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
            num_filters = self.num_filters

            mt_x = data.mt_x - 1

            embedding = self.embedding(mt_x)
            embedding = embedding.squeeze()

            inter_item_emb = self.group_graph.forward(embedding, data)

            # final_s = self.fuse_model.forward(intra_item_emb, inter_item_emb, data.sequence_len)
            # final_s = self.narm.forward(intra_item_emb, inter_item_emb, data.sequence_len)
            final_s = self.cnn_fusing.forward(intra_item_emb, inter_item_emb, data.sequence_len)

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

