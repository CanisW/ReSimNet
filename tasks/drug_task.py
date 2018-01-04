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
        self.key_set = self.process_data(data_path)
        self.dataset = self.load_pairs(self.key_set)

    def initial_setting(self):
        # Dataset split into train/valid/test
        self.key_set = []
        self.dataset = {'tr': [], 'va': [], 'te': []} # TODO
        self.input_maxlen = 0

        # Static values
        self.FEATURE_NUM = 8
        self.INCHI_KEY_IDX = 5
        self.KEY_LENGTH = 27

        # Key dictionaries
        self.key2idx = {'a': {}, 'b': {}}
        self.idx2key = {'a': {}, 'b': {}}

        # Character dictionaries
        self.char2idx = {}
        self.idx2char = {}
    
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
        key_set = []
        # pairs = {'tr': [], 'va': [], 'te': []}

        with open(path) as f:
            csv_reader = csv.reader(f)
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    continue
                
                # Skip invalid rows, keys
                key = row[self.INCHI_KEY_IDX]
                assert self.FEATURE_NUM == len(row), len(row)
                if len(key.split('-')) != 3:
                    print('Invalid key: ', key)
                    continue
                assert self.KEY_LENGTH == len(key), len(key)

                # Register each key to dictionaries
                key_a, key_b, _ = key.split('-')
                self.register_key_a(key_a)
                self.register_key_b(key_b)
                list(map(lambda x: self.register_char(x), key))

                key_set.append([list(map(lambda x: self.char2idx[x], key)),
                    self.key2idx['a'][key_a], self.key2idx['b'][key_b]])

        print('Character dictionary size {}'.format(len(self.char2idx)))
        print('Key A dictionary size {}'.format(len(self.key2idx['a'])))
        print('Key B dictionary size {}'.format(len(self.key2idx['b'])))
        print('Dataset size {}\n'.format(len(key_set)))

        return np.array(key_set)
    
    def load_pairs(self, key_set):
        raise NotImplementedError

    def loader(self, batch_size=16, maxlen=None):
        if maxlen is None: 
            maxlen = float("inf")

        batch_key1 = []
        batch_key1a = []
        batch_key1b = []
        batch_key2 = []
        batch_key2a = []
        batch_key2b = []
        batch_sim = []

        for idx in range(0, self.dataset_len):
            key_idxs = random.sample(range(len(self.key_set)), 2)
            key1, key2 = self.key_set[key_idxs, :]
            batch_key1.append(key1[0])
            batch_key1a.append(key1[1])
            batch_key1b.append(key1[2])
            batch_key2.append(key2[0])
            batch_key2a.append(key2[1])
            batch_key2b.append(key2[2])
            batch_sim.append(np.random.uniform(-1, 1, size=1))

            if len(batch_key1) == batch_size:
                yield (batch_key1, batch_key1a, batch_key1b, batch_key2, 
                       batch_key2a, batch_key2b, batch_sim)
                del (batch_key1[:], batch_key1a[:], batch_key1b[:], 
                     batch_key2[:], batch_key2a[:], batch_key2b[:], batch_sim[:])

    def shuffle_data(self):
        d = self.dataset[self._mode]
        zipped = list(zip(d[0], d[1], d[2], d[3], d[4]))
        random.shuffle(zipped)
        d[0], d[1], d[2], d[3], d[4] = zip(*zipped)

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
        if self._mode == 'tr':
            return 16*1000
        elif self._mode == 'va':
            return 16*100
        elif self._mode == 'te':
            return 16*100


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
    data_path = './data/drug/inchikey_info.csv'
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
        dataset.decode_data(k1[0], k1a[0], k1b[0], k2[0], k2a[0], k2b[0], 
                sim[0])

