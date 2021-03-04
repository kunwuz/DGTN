import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import pickle
import time

raw_path = '../datasets/raw/cikm16-train-item-views'
save_path = "../neigh_retrieval/unaugment_data"


def load_data(file):
    print("Start load_data")
    # load csv
    data = pd.read_csv(file+'.csv', sep=';', header=0, usecols=[0, 2, 4], dtype={0: np.int32, 1: np.int64, 3: str})
    # specify header names
    data.columns = ['SessionId', 'ItemId', 'Eventdate']
    # convert time string to timestamp and remove the original column
    data['Time'] = data.Eventdate.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    print(data['Time'].min())
    print(data['Time'].max())
    del(data['Eventdate'])

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data


def filter_data(data, min_item_support=5, min_session_length=2):
    print("Start filter_data")

    # y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]

    # filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= min_item_support].index)]

    # filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    print(data['Time'].min())
    print(data['Time'].max())
    # output
    data_start = datetime.fromtimestamp(data.Time.astype(np.int64).min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.astype(np.int64).max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    return data


def split_train_test(data):
    print("Start split_train_test")
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-7*86400].index
    session_test = session_max_times[session_max_times >= tmax-7*86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))

    return train, test


def get_dict(data):
    print("Start get_dict")
    item2idx = {}
    pop_scores = data.groupby('ItemId').size().sort_values(ascending=False)
    pop_scores = pop_scores / pop_scores[:1].values[0]
    items = pop_scores.index
    for idx, item in enumerate(items):
        item2idx[item] = idx+1

    return item2idx


def process_seqs(seqs, shift):
    start = time.time()
    labs = []
    index = shift
    for count, seq in enumerate(seqs):
        index += (len(seq) - 1)
        labs += [index]
        end = time.time()
        print("\rprocess_seqs: [%d/%d], %.2f, usetime: %fs, " % (count, len(seqs), count/len(seqs) * 100, end - start),
              end='', flush=True)
    print("\n")
    return seqs, labs


def get_sequence(data, item2idx, shift=-1):
    start = time.time()
    sess_ids = data.drop_duplicates('SessionId', 'first')
    print(sess_ids)
    sess_ids.sort_values(['Time'], inplace=True)
    sess_ids = sess_ids['SessionId'].unique()
    seqs = []
    for count, sess_id in enumerate(sess_ids):
        seq = data[data['SessionId'].isin([sess_id])]
        seq = seq['ItemId'].values
        outseq = []
        for i in seq:
            if i in item2idx:
                outseq += [item2idx[i]]
        seqs += [outseq]
        end = time.time()
        print("\rGet_sequence: [%d/%d], %.2f , usetime: %fs" % (count, len(sess_ids), count/len(sess_ids) * 100, end - start),
              end='', flush=True)

    print("\n")
    out_seqs, labs = process_seqs(seqs, shift)
    print(len(out_seqs), len(labs))
    return out_seqs, labs


def preprocess(train, test, path=save_path):
    print("--------------")
    print("Start preprocess cikm16")
    item2idx = get_dict(train)
    train_seqs, train_labs = get_sequence(train, item2idx)
    test_seqs, test_labs = get_sequence(test, item2idx, train_labs[-1])
    train = (train_seqs, train_labs)
    test = (test_seqs, test_labs)

    path = path + '/cikm16'
    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(test, open(path+'/unaug_test.txt', 'wb'))
    pickle.dump(train, open(path+'/unaug_train.txt', 'wb'))
    print("finished")


data = load_data(raw_path)
data = filter_data(data)
train, test = split_train_test(data)
preprocess(train, test)
