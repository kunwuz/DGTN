# -*- coding: utf-8 -*-
"""
Created on 6/6/2019
@author: RuihongQiu
"""


import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform


class InOutGGNN(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, out_channels, num_layers, aggr='add', bias=True):
        super(InOutGGNN, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, 2, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(2 * out_channels, out_channels, bias=bias)
        self.bias_in = Param(Tensor(self.out_channels))
        self.bias_out = Param(Tensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=[None, None]):
        #print(edge_weight[0].size(), edge_weight[1].size)

        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            self.flow = 'source_to_target'
            h1 = torch.matmul(h, self.weight[i, 0])
            m1 = self.propagate(edge_index, x=h1, edge_weight=edge_weight[0], bias=self.bias_in)
            self.flow = 'target_to_source'
            h2 = torch.matmul(h, self.weight[i, 1])
            m2 = self.propagate(edge_index, x=h2, edge_weight=edge_weight[1], bias=self.bias_out)
            h = self.rnn(torch.cat((m1, m2), dim=-1), h)

        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out, bias):
        if bias is not None:
            return aggr_out + bias
        else:
            return aggr_out

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)
