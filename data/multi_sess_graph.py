import time
import pickle
import torch
import collections
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, Dataset


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, knn_phrase, transform=None, pre_transform=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
        self.knn_phrase = knn_phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']

    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']

    def download(self):
        pass

    def find_neighs(self, index, knn_data):
        sess_neighs = knn_data[index]
        if sess_neighs == 0:
            return []
        else:
            return list(np.asarray(sess_neighs).astype(np.int32))

    def multi_process(self, train_data, knn_data, sess_index, y):
        # find neigh
        neigh_index = self.find_neighs(sess_index, knn_data)
        # neigh_index = []
        neigh_index.append(sess_index)
        temp_neighs = train_data[neigh_index]
        neighs = []

        # append y
        for neigh, idx in zip(temp_neighs, neigh_index):
            if idx != sess_index:
                neigh.append(y[idx])
            neighs.append(neigh)

        nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
        all_senders = []
        all_receivers = []
        x = []
        i = 0
        for sess in neighs:
            senders = []
            for node in sess:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]

            if len(senders) != 1:
                del senders[-1]  # the last item is a receiver
                del receivers[0]  # the first item is a sender
            all_senders += senders
            all_receivers += receivers

        sess = train_data[sess_index]
        sess_item_index = [nodes[item] for item in sess]
        # num_count = [count[i[0]] for i in x]

        sess_masks = np.zeros(len(nodes))
        sess_masks[sess_item_index] = 1

        pair = {}
        sur_senders = all_senders[:]
        sur_receivers = all_receivers[:]
        i = 0
        for sender, receiver in zip(sur_senders, sur_receivers):
            if str(sender) + '-' + str(receiver) in pair:
                pair[str(sender) + '-' + str(receiver)] += 1
                del all_senders[i]
                del all_receivers[i]
            else:
                pair[str(sender) + '-' + str(receiver)] = 1
                i += 1

        node_num = len(x)

        # num_count = torch.tensor(num_count, dtype=torch.float)
        edge_index = torch.tensor([all_senders, all_receivers], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.long)
        node_num = torch.tensor([node_num], dtype=torch.long)
        sess_item_idx = torch.tensor(sess_item_index, dtype=torch.long)
        sess_masks = torch.tensor(sess_masks, dtype=torch.long)

        return x, edge_index, node_num, sess_item_idx, sess_masks

    def single_process(self, sequence, y):
        # sequence = [1, 2, 3, 2, 4]
        count = collections.Counter(sequence)
        i = 0
        nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
        senders = []
        x = []
        for node in sequence:
            if node not in nodes:
                nodes[node] = i
                x.append([node])
                i += 1
            senders.append(nodes[node])
        receivers = senders[:]
        num_count = [count[i[0]] for i in x]

        sess_item_index = [nodes[item] for item in sequence]

        if len(senders) != 1:
            del senders[-1]  # the last item is a receiver
            del receivers[0]  # the first item is a sender

        pair = {}
        sur_senders = senders[:]
        sur_receivers = receivers[:]
        i = 0
        for sender, receiver in zip(sur_senders, sur_receivers):
            if str(sender) + '-' + str(receiver) in pair:
                pair[str(sender) + '-' + str(receiver)] += 1
                del senders[i]
                del receivers[i]
            else:
                pair[str(sender) + '-' + str(receiver)] = 1
                i += 1

        count = collections.Counter(senders)
        out_degree_inv = [1 / count[i] for i in senders]

        count = collections.Counter(receivers)
        in_degree_inv = [1 / count[i] for i in receivers]

        in_degree_inv = torch.tensor(in_degree_inv, dtype=torch.float)
        out_degree_inv = torch.tensor(out_degree_inv, dtype=torch.float)

        edge_count = [pair[str(senders[i]) + '-' + str(receivers[i])] for i in range(len(senders))]
        edge_count = torch.tensor(edge_count, dtype=torch.float)

        # senders, receivers = senders + receivers, receivers + senders

        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor([y], dtype=torch.long)
        num_count = torch.tensor(num_count, dtype=torch.float)
        sequence = torch.tensor(sequence, dtype=torch.long)
        sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
        sess_item_idx = torch.tensor(sess_item_index, dtype=torch.long)


        return x, y, num_count, edge_index, edge_count, sess_item_idx, sequence_len, in_degree_inv, out_degree_inv

    def process(self):
        start = time.time()
        train_data = pickle.load(open(self.raw_dir + '/' + 'train.txt', 'rb'))
        test_data = pickle.load(open(self.raw_dir + '/' + 'test.txt', 'rb'))
        # knn_data = np.load(self.raw_dir + '/' + self.knn_phrase + '.npy')
        knn_data = pickle.load(open(self.raw_dir + '/' + self.knn_phrase + '.txt', "rb"))
        data_list = []
        if self.phrase == "train":
            sess_index = 0
            data = train_data
            total_data = np.asarray(train_data[0])
            total_label = np.asarray(train_data[1])
        else:
            sess_index = len(train_data[0])
            data = test_data
            total_data = np.concatenate((train_data[0], test_data[0]), axis=0)
            total_label = np.concatenate((train_data[1], test_data[1]), axis=0)

        for sequence, y in zip(data[0], data[1]):

            mt_x, mt_edge_index, mt_node_num, mt_sess_item_idx, sess_masks = \
                self.multi_process(total_data, knn_data, sess_index, total_label)

            x, y, num_count, edge_index, edge_count, sess_item_idx, sequence_len, in_degree_inv, out_degree_inv = \
                self.single_process(sequence, y)

            session_graph = Data(x=x, y=y, num_count=num_count, sess_item_idx=sess_item_idx,
                                    edge_index=edge_index, edge_count=edge_count, sequence_len=sequence_len,
                                    in_degree_inv=in_degree_inv, out_degree_inv=out_degree_inv,
                                    mt_x=mt_x, mt_edge_index=mt_edge_index, mt_node_num=mt_node_num,
                                    mt_sess_item_idx=mt_sess_item_idx, sess_masks=sess_masks)

            data_list.append(session_graph)
            sess_index += 1

            end = time.time()
            if sess_index % (len(data[0]) // 1000) == 0:
                print("\rProcess_seqs: [%d/%d], %.2f, usetime: %fs, " % (sess_index, len(data[0]), sess_index/len(data[0]) * 100, end - start),
              end='', flush=True)
        print('\nStart collate')
        data, slices = self.collate(data_list)
        print('\nStart save')
        torch.save((data, slices), self.processed_paths[0])

