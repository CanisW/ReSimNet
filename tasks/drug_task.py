import numpy as np
import sys
import copy
import pickle
import string
import os
import random
import csv
from os.path import expanduser


class DrugDataset(object):
    def __init__(self, data_path):
        self.initial_setting()
        self.pairs = self.process_data(data_path)
        self.dataset = self.split_dataset(self.pairs)

    def initial_setting(self):
        # Dataset split into train/valid/test
        self.pairs = []
        self.dataset = {'tr': [], 'va': [], 'te': []}
        self.SR = [0.7, 0.1, 0.2] # split raio

        # Static values
        self.FEATURE_NUM = 11
        self.ORG_INCHI_IDX = 4
        self.TAR_INCHI_IDX = 5
        self.KEY_LENGTH = 27
        self.SIM_IDX = 8
        self.NUM_SIM = 3

        # Key dictionaries (a part, b part, total)
        self.key2idx = {'a': {}, 'b': {}, 't': {}}
        self.idx2key = {'a': {}, 'b': {}, 't': {}}

        # Character dictionaries
        self.char2idx = {}
        self.idx2char = {}

    def register_key(self, key):
        if key not in self.key2idx['t']:
            self.key2idx['t'][key] = len(self.key2idx['t'])
            self.idx2key['t'][len(self.idx2key['t'])] = key
    
    def register_key_a(self, key_a):
        if key_a not in self.key2idx['a']:
            self.key2idx['a'][key_a] = len(self.key2idx['a'])
            self.idx2key['a'][len(self.idx2key['a'])] = key_a

    def register_key_b(self, key_b):
        if key_b not in self.key2idx['b']:
            self.key2idx['b'][key_b] = len(self.key2idx['b'])
            self.idx2key['b'][len(self.idx2key['b'])] = key_b

    def register_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = len(self.char2idx)
            self.idx2char[len(self.idx2char)] = char

    def process_data(self, path):
        print('### Processing {}'.format(path))
        key1_set = []
        key1_a_set = []
        key1_b_set = []
        key2_set = []
        key2_a_set = []
        key2_b_set = []
        similarities = []

        with open(path) as f:
            csv_reader = csv.reader(f)
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    continue
                
                # Skip invalid rows, keys
                key1 = row[self.ORG_INCHI_IDX]
                key2 = row[self.TAR_INCHI_IDX]
                similarity = row[self.SIM_IDX:self.SIM_IDX+self.NUM_SIM]
                similarity = list(map(lambda x: float(x), similarity))
                assert self.FEATURE_NUM == len(row), len(row)
                assert len(key1.split('-')) == 3
                assert len(key2.split('-')) == 3
                assert key1.split('-')[2] == key2.split('-')[2] == 'N'
                assert self.KEY_LENGTH == len(key1), len(key)
                assert self.KEY_LENGTH == len(key2), len(key)

                # Register each key to dictionaries
                key1_a, key1_b, _ = key1.split('-')
                key2_a, key2_b, _ = key2.split('-')
                self.register_key(key1)
                self.register_key(key2)
                self.register_key_a(key1_a)
                self.register_key_b(key1_b)
                self.register_key_a(key2_a)
                self.register_key_b(key2_b)
                list(map(lambda x: self.register_char(x), key1))
                list(map(lambda x: self.register_char(x), key2))

                # Save each key and similarities
                key1_set.append(list(map(lambda x: self.char2idx[x], key1)))
                key1_a_set.append(self.key2idx['a'][key1_a])
                key1_b_set.append(self.key2idx['b'][key1_b])
                key2_set.append(list(map(lambda x: self.char2idx[x], key2)))
                key2_a_set.append(self.key2idx['a'][key2_a])
                key2_b_set.append(self.key2idx['b'][key2_b])
                similarities.append(similarity)

        pairs = [key1_set, key1_a_set, key1_b_set,
                 key2_set, key2_a_set, key2_b_set, similarities]

        print('Key dictionary size {}'.format(len(self.key2idx['t'])))
        print('Key A dictionary size {}'.format(len(self.key2idx['a'])))
        print('Key B dictionary size {}'.format(len(self.key2idx['b'])))
        print('Character dictionary size {}'.format(len(self.char2idx)))
        print('Dataset size {}'.format(len(key1_set)))

        return pairs
    
    def split_dataset(self, p):
        zipped = list(zip(p[0], p[1], p[2], p[3], p[4], p[5], p[6]))
        random.shuffle(zipped)
        shuffled_p = list(zip(*zipped))

        tr_idx, va_idx, te_idx = list(map(lambda x: int(x*len(p[0])), self.SR))
        train = list(map(
            lambda x: list(x)[:tr_idx], shuffled_p))
        valid = list(map(
            lambda x: x[tr_idx:tr_idx+va_idx], shuffled_p))
        test = list(map(
            lambda x: x[tr_idx+va_idx:], shuffled_p))

        print('Train/Valid/Test split: {}/{}/{}\n'.format(
              len(train[0]), len(valid[0]), len(test[0])))

        return {'tr': train, 'va': valid, 'te': test}

    def loader(self, batch_size=16, sim_idx=0):
        batch_key1 = []
        batch_key1a = []
        batch_key1b = []
        batch_key2 = []
        batch_key2a = []
        batch_key2b = []
        batch_sim = []
        d = self.dataset[self._mode]

        for d0, d1, d2, d3, d4, d5, d6 \
                in zip(d[0], d[1], d[2], d[3], d[4], d[5], d[6]):
            # Make it cleaner
            d0 = np.array(d0)
            d1 = np.array(d1)
            d2 = np.array(d2)
            d3 = np.array(d3)
            d4 = np.array(d4)
            d5 = np.array(d5)
            d6 = np.array(d6)
            batch_key1.append(d0)
            batch_key1a.append(d1)
            batch_key1b.append(d2)
            batch_key2.append(d3)
            batch_key2a.append(d4)
            batch_key2b.append(d5)
            batch_sim.append(d6[sim_idx])

            if len(batch_key1) == batch_size:
                yield (batch_key1, batch_key1a, batch_key1b, batch_key2, 
                       batch_key2a, batch_key2b, batch_sim)
                del (batch_key1[:], batch_key1a[:], batch_key1b[:], 
                     batch_key2[:], batch_key2a[:], batch_key2b[:], batch_sim[:])

    def shuffle_data(self):
        d = self.dataset[self._mode]
        zipped = list(zip(d[0], d[1], d[2], d[3], d[4], d[5], d[6]))
        random.shuffle(zipped)
        d[0], d[1], d[2], d[3], d[4], d[5], d[6] = zip(*zipped)

    def decode_data(self, k1, k1a, k1b, k2, k2a, k2b, sim):
        print('Key1: {}'.format(''.join(list(map(
            lambda x: self.idx2char[x], k1)))))
        print('Key1 A: {}'.format(self.idx2key['a'][k1a]))
        print('Key1 B: {}'.format(self.idx2key['b'][k1b]))
        print('Key2: {}'.format(''.join(list(map(
            lambda x: self.idx2char[x], k2)))))
        print('Key2 A: {}'.format(self.idx2key['a'][k2a]))
        print('Key2 B: {}'.format(self.idx2key['b'][k2b]))
        print('Similarity: {}\n'.format(sim))

    # Dataset mode ['tr', 'va']
    def set_mode(self, mode):
        self._mode = mode

    @property
    def dataset_len(self):
        return len(self.dataset[self._mode][0])


"""
[Version Note]
    v0.1: basic implementation
        key A, key B, char: 9165/5677/27 
        key set: 20337
        train:
        valid: 
        test: 
    
"""

if __name__ == '__main__':

    # Dataset configuration 
    data_path = './data/drug/pert_df_id_centroid_pair.csv'
    save_preprocess = True
    save_path = './data/drug/drug(tmp).pkl'
    load_path = './data/drug/drug(tmp).pkl'

    # Save or load dataset
    if save_preprocess:
        dataset = DrugDataset(data_path)
        pickle.dump(dataset, open(save_path, 'wb'))
        print('## Save preprocess %s' % load_path)
    else:
        print('## Load preprocess %s' % load_path)
        dataset = pickle.load(open(load_path, 'rb'))
   
    # Loader testing
    dataset.set_mode('va')
    # dataset.shuffle_data()

    for idx, (k1, k1a, k1b, k2, k2a, k2b, sim) in enumerate(dataset.loader()):
        k1, k1a, k1b, k2, k2a, k2b, sim  = (np.array(xx) for xx in [k1, k1a, k1b,
                                            k2, k2a, k2b, sim])
        # dataset.decode_data(k1[0], k1a[0], k1b[0], k2[0], k2a[0], k2b[0], 
        #         sim[0])
        pass

