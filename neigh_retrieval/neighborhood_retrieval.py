import numpy as np
import pickle
from knn import KNN


dataset = 'yoochoose1_64/'
save_path = '../datasets/' + dataset + 'raw/'

org_test_data = pickle.load(open('../datasets/' + dataset + 'raw/test.txt', 'rb'))
org_train_data = pickle.load(open('../datasets/' + dataset + 'raw/train.txt', 'rb'))
unaug_test_data = pickle.load(open('unaugment_data/' + dataset + '/unaug_test.txt', 'rb'))
unaug_train_data = pickle.load(open('unaugment_data/' + dataset + '/unaug_train.txt', 'rb'))

test_data = org_test_data[0]
train_data = org_train_data[0]
all_data = np.concatenate((train_data, test_data), axis=0)

unaug_data = np.concatenate((unaug_train_data[0], unaug_test_data[0]), axis=0)
unaug_index = np.concatenate((unaug_train_data[1], unaug_test_data[1]), axis=0)

del org_test_data, org_train_data
del test_data, train_data
del unaug_train_data, unaug_test_data

k_num = [20,40,60,100,140, 160, 180, 200]

for k in k_num:
    knn = KNN(k, all_data, unaug_data, unaug_index)
    all_sess_neigh = knn.get_neigh_sess(0)
    pickle.dump(all_sess_neigh, open(save_path+"neigh_data_"+str(k)+".txt", "wb"))
    lens = 0
    for i in all_sess_neigh:
        if i != 0:
            lens += len(i)
    print(lens / len(all_sess_neigh))
