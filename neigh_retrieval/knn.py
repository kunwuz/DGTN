import numpy as np
import math
import time

class KNN:
    def __init__(self, k, all_sess, unaug_data, unaug_index, threshold=0.5, samples=1000):
        self.k = k
        self.all_sess = all_sess
        self.threshold = threshold
        self.samples = samples
        self.item_sess_map = self.get_item_sess_map(unaug_index, unaug_data)
        self.no_pro_data = unaug_data
        self.no_pro_index = unaug_index


    def get_item_sess_map(self, unaug_index, unaug_data):
        item_sess_map = {}
        for index, sess in zip(unaug_index, unaug_data):
            items = np.unique(sess[:-1])
            for item in items:
                if item not in item_sess_map.keys():
                    item_sess_map[item] = []
                item_sess_map[item].append(index)
        print("get_item_sess_map over")
        return item_sess_map

    def jaccard(self, first, second):

        intersection = len(set(first).intersection(set(second)))
        union = len(set(first).union(set(second)))
        res = intersection / union

        return res

    def cosine(self, first, second):

        li = len(set(first).intersection(set(second)))
        la = len(first)
        lb = len(second)
        result = li / (math.sqrt(la) * math.sqrt(lb))

        return result

    def vec(self, first, second, pos_map):
        a = set(first).intersection(set(second))
        sum = 0
        for i in a:
            sum += pos_map[i]

        result = sum / len(pos_map)

        return result

    def find_sess(self, sess, item_sess_map):
        items = np.unique(sess)
        sess_index = []
        for item in items:
            sess_index += item_sess_map[item]
        return sess_index

    def calc_similarity(self, target_session, all_data, sess_index):
        neighbors = []
        session_items = np.unique(target_session)

        possible_sess_index = self.find_sess(session_items, self.item_sess_map)
        possible_sess_index = [p_index for p_index in possible_sess_index if p_index < sess_index]
        possible_sess_index = sorted(np.unique(possible_sess_index))[-self.samples:]
        possible_sess_index = sorted(np.unique(possible_sess_index))

        pos_map = {}
        length = len(target_session)

        count = 1
        for item in target_session:
            pos_map[item] = count / length
            count += 1

        for index in possible_sess_index:
            session = all_data[index]
            session_items_test = np.unique(session)
            similarity = np.around(self.cosine(session_items_test, session_items), 4)
            if similarity >= self.threshold:
                neighbors.append([index, similarity])

        return neighbors

    def get_neigh_sess(self, index):
        all_sess_neigh = []
        start = time.time()
        all_sess = self.all_sess[index:]
        for sess in all_sess:
            possible_neighbors = self.calc_similarity(sess, self.all_sess, index)
            possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])

            if len(possible_neighbors) > 0:
                possible_neighbors = list(np.asarray(possible_neighbors)[:, 0])
            if len(possible_neighbors) > self.k:
                all_sess_neigh.append(possible_neighbors[:self.k])
            elif len(possible_neighbors) > 0:
                all_sess_neigh.append(possible_neighbors)
            else:
                all_sess_neigh.append(0)
            index += 1
            end = time.time()

            if index % (len(self.all_sess) // 100) == 0:
                print("\rProcess_seqs: [%d/%d], %.2f, usetime: %fs, " % (index, len(self.all_sess), index/len(self.all_sess) * 100, end - start),
              end='', flush=True)

        return all_sess_neigh
