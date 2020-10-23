import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

import math


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class InOutGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 concat=False,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 middle_layer=False,
                 **kwargs):
        super(InOutGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.middle_layer = middle_layer
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight1 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.weight2 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if concat and not middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels * heads, bias=bias)
        elif middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels, bias=bias)
        else:
            self.rnn = torch.nn.GRUCell(2 * out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, sess_masks):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        sess_masks = sess_masks.view(sess_masks.shape[0], 1).float()
        xs = x * sess_masks
        xns = x * (1 - sess_masks)

        # self.flow = 'source_to_target'
        # x1 = torch.mm(x, self.weight[0]).view(-1, self.heads, self.out_channels)
        # m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        # self.flow = 'target_to_source'
        # x2 = torch.mm(x, self.weight[1]).view(-1, self.heads, self.out_channels)
        # m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        self.flow = 'source_to_target'
        x1s = torch.mm(xs, self.weight1[0]).view(-1, self.heads, self.out_channels)
        print(x1s.shape())
        x1ns = torch.mm(xns, self.weight2[0]).view(-1, self.heads, self.out_channels)
        print(x1ns.shape())
        x1 = x1s + x1ns
        m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        self.flow = 'target_to_source'
        x2s = torch.mm(xs, self.weight1[1]).view(-1, self.heads, self.out_channels)
        x2ns = torch.mm(xns, self.weight2[1]).view(-1, self.heads, self.out_channels)
        x2 = x2s + x2ns
        m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        if not self.middle_layer:
            if self.concat:
                x = x.repeat(1, self.heads)
            else:
                x = x.view(-1, self.heads, self.out_channels).mean(dim=1)

        # x = self.rnn(torch.cat((m1, m2), dim=-1), x)
        x = m1 + m2
        # x = m1
        return x

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class InOutGATConv_intra(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 middle_layer=False,
                 **kwargs):
        super(InOutGATConv_intra, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.middle_layer = middle_layer
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.weight1 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.weight2 = Parameter(
            torch.Tensor(2, in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if concat and not middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels * heads, bias=bias)
        elif middle_layer:
            self.rnn = torch.nn.GRUCell(2 * out_channels * heads, in_channels, bias=bias)
        else:
            self.rnn = torch.nn.GRUCell(2 * out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, sess_masks):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # sess_masks = sess_masks.view(sess_masks.shape[0], 1).float()
        # xs = x * sess_masks
        # xns = x * (1 - sess_masks)

        self.flow = 'source_to_target'
        x1 = torch.mm(x, self.weight[0]).view(-1, self.heads, self.out_channels)
        m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        self.flow = 'target_to_source'
        x2 = torch.mm(x, self.weight[1]).view(-1, self.heads, self.out_channels)
        m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        # self.flow = 'source_to_target'
        # x1s = torch.mm(xs, self.weight1[0]).view(-1, self.heads, self.out_channels)
        # x1ns = torch.mm(xns, self.weight2[0]).view(-1, self.heads, self.out_channels)
        # x1 = x1s + x1ns
        # m1 = self.propagate(edge_index, x=x1, num_nodes=x.size(0))
        # self.flow = 'target_to_source'
        # x2s = torch.mm(xs, self.weight1[1]).view(-1, self.heads, self.out_channels)
        # x2ns = torch.mm(xns, self.weight2[1]).view(-1, self.heads, self.out_channels)
        # x2 = x2s + x2ns
        # m2 = self.propagate(edge_index, x=x2, num_nodes=x.size(0))

        if not self.middle_layer:
            if self.concat:
                x = x.repeat(1, self.heads)
            else:
                x = x.view(-1, self.heads, self.out_channels).mean(dim=1)

        # x = self.rnn(torch.cat((m1, m2), dim=-1), x)
        x = m1 + m2
        return x

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)