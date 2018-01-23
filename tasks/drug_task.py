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
        self.pairs = self.pad_sequence(self.pairs)
        self.dataset = self.split_dataset(self.pairs)

    def initial_setting(self):
        # Dataset split into train/valid/test
        self.pairs = []
        self.dataset = {'tr': [], 'va': [], 'te': []}
        self.SR = [0.7, 0.1, 0.2] # split ratio
        self.UR = 0.1 # Unknown ratio
        self.input_maxlen = 0

        # Static values
        self.FEATURE_NUM = 8
        self.ORG_INCHI_IDX = 3
        self.TAR_INCHI_IDX = 4
        self.SIM_IDX = 7
        self.NUM_SIM = 1

        # Key dictionaries
        self.key2idx = {}
        self.idx2key = {}
        self.known = {}
        self.unknown = {}
        self.PAD = 'PAD'

        # Character dictionaries
        self.char2idx = {}
        self.idx2char = {}

    def register_key(self, key):
        if key not in self.key2idx:
            self.key2idx[key] = len(self.key2idx)
            self.idx2key[len(self.idx2key)] = key

    def register_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = len(self.char2idx)
            self.idx2char[len(self.idx2char)] = char

    def process_data(self, path):
        print('### Processing {}'.format(path))
        key1_set = []
        key1_len = []
        key2_set = []
        key2_len = []
        similarities = []
        self.register_char(self.PAD)

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

                # Register each key to dictionaries
                self.register_key(key1)
                self.register_key(key2)
                list(map(lambda x: self.register_char(x), key1))
                list(map(lambda x: self.register_char(x), key2))

                # Update maxlen
                self.input_maxlen = self.input_maxlen if self.input_maxlen > \
                        len(key1) else len(key1)
                self.input_maxlen = self.input_maxlen if self.input_maxlen > \
                        len(key2) else len(key2)

                # Save each key and similarities
                key1_set.append(list(map(lambda x: self.char2idx[x], key1)))
                key1_len.append(len(key1))
                key2_set.append(list(map(lambda x: self.char2idx[x], key2)))
                key2_len.append(len(key2))
                similarities.append(similarity)

        pairs = [key1_set, key1_len, key2_set, key2_len, similarities]

        print('Key dictionary size {}'.format(len(self.key2idx)))
        print('Character dictionary size {}'.format(len(self.char2idx)))
        print('Input maxlen {}'.format(self.input_maxlen))
        print('Dataset size {}'.format(len(key1_set)))

        return pairs
    
    def pad_sequence(self, p):
        # Pad key1_set
        for key1 in p[0]:
            while len(key1) != self.input_maxlen:
                key1.append(self.char2idx[self.PAD])
            assert len(key1) == self.input_maxlen

        # Pad key2_set
        for key2 in p[2]:
            while len(key2) != self.input_maxlen:
                key2.append(self.char2idx[self.PAD])
            assert len(key2) == self.input_maxlen

        return p
    
    def split_dataset(self, p):
        # Shuffle key dicitonary
        items = list(self.key2idx.items())
        random.shuffle(items)
        self.known = dict(items[:int(-len(items) * self.UR)])
        self.unknown = dict(items[int(-len(items) * self.UR):])

        # Unknown check
        for unk, _ in self.unknown.items():
            assert unk not in self.known

        # Shuffle dataset
        zipped = list(zip(p[0], p[1], p[2], p[3], p[4]))
        random.shuffle(zipped)
        sf_p = list(zip(*zipped))

        # Ready for train/valid/test
        train = [[], [], [], [], []]
        valid = [[], [], [], [], []]
        test = [[], [], [], [], []]
        valid_kk = 0
        valid_ku = 0
        valid_uu = 0
        test_kk = 0
        test_ku = 0
        test_uu = 0

        # Iterate and split by unknown ratio
        for p0, p1, p2, p3, p4 in zip(sf_p[0], sf_p[1], 
                                      sf_p[2], sf_p[3], sf_p[4]):
            key1 = ''.join(list(map(lambda x: self.idx2char[x], p0[:p1])))
            key2 = ''.join(list(map(lambda x: self.idx2char[x], p2[:p3])))

            if key1 in self.unknown or key2 in self.unknown:
                is_test = np.random.binomial(1, 2/3.)
                if is_test:
                    test[0].append(p0)
                    test[1].append(p1)
                    test[2].append(p2)
                    test[3].append(p3)
                    test[4].append(p4)
                    if key1 in self.unknown and key2 in self.unknown:
                        test_uu += 1
                    else:
                        test_ku += 1
                else:
                    valid[0].append(p0)
                    valid[1].append(p1)
                    valid[2].append(p2)
                    valid[3].append(p3)
                    valid[4].append(p4)
                    if key1 in self.unknown and key2 in self.unknown:
                        valid_uu += 1
                    else:
                        valid_ku += 1

        # Fill known/known set with limit of split ratio
        for p0, p1, p2, p3, p4 in zip(sf_p[0], sf_p[1], 
                                      sf_p[2], sf_p[3], sf_p[4]):
            key1 = ''.join(list(map(lambda x: self.idx2char[x], p0[:p1])))
            key2 = ''.join(list(map(lambda x: self.idx2char[x], p2[:p3])))

            if key1 not in self.unknown and key2 not in self.unknown:
                assert key1 in self.known and key2 in self.known
                if len(train[0]) < len(sf_p[0]) * self.SR[0]:
                    train[0].append(p0)
                    train[1].append(p1)
                    train[2].append(p2)
                    train[3].append(p3)
                    train[4].append(p4)
                elif len(valid[0]) < len(sf_p[0]) * self.SR[1]: 
                    valid[0].append(p0)
                    valid[1].append(p1)
                    valid[2].append(p2)
                    valid[3].append(p3)
                    valid[4].append(p4)
                    valid_kk += 1
                else:
                    test[0].append(p0)
                    test[1].append(p1)
                    test[2].append(p2)
                    test[3].append(p3)
                    test[4].append(p4)
                    test_kk += 1

        print('Train/Valid/Test split: {}/{}/{}'.format(
              len(train[0]), len(valid[0]), len(test[0])))
        print('Valid/Test KK,KU,UU: ({},{},{})/({},{},{})\n'.format(
              valid_kk, valid_ku, valid_uu, test_kk, test_ku, test_uu))

        return {'tr': train, 'va': valid, 'te': test}

    def loader(self, batch_size=16, sim_idx=0):
        batch_key1 = []
        batch_key1_len = []
        batch_key2 = []
        batch_key2_len = []
        batch_sim = []
        d = self.dataset[self._mode]

        for d0, d1, d2, d3, d4 in zip(d[0], d[1], d[2], d[3], d[4]):
            # Make it cleaner
            d0 = np.array(d0)
            d1 = np.array(d1)
            d2 = np.array(d2)
            d3 = np.array(d3)
            d4 = np.array(d4)
            batch_key1.append(d0)
            batch_key1_len.append(d1)
            batch_key2.append(d2)
            batch_key2_len.append(d3)
            batch_sim.append(d4[sim_idx])

            if len(batch_key1) == batch_size:
                yield (batch_key1, batch_key1_len, 
                       batch_key2, batch_key2_len, batch_sim)
                del (batch_key1[:], batch_key1_len[:],
                     batch_key2[:], batch_key2_len[:], batch_sim[:])

        yield (batch_key1, batch_key1_len, 
               batch_key2, batch_key2_len, batch_sim)

    def shuffle(self):
        d = self.dataset[self._mode]
        zipped = list(zip(d[0], d[1], d[2], d[3], d[4]))
        random.shuffle(zipped)
        d[0], d[1], d[2], d[3], d[4] = zip(*zipped)

    def decode_data(self, k1, k1_l, k2, k2_l, sim):
        print('Key1: {}, length: {}'.format(''.join(list(map(
            lambda x: self.idx2char[x], k1[:k1_l]))), k1_l))
        print('Key2: {}, length: {}'.format(''.join(list(map(
            lambda x: self.idx2char[x], k2[:k2_l]))), k2_l))
        print('Similarity: {}\n'.format(sim))

    # Dataset mode ['tr', 'va']
    def set_mode(self, mode):
        self._mode = mode

    @property
    def length(self):
        return len(self.dataset[self._mode][0])


"""
[Version Note]
    v0.1: basic implementation
        key A, key B, char: 9165/5677/27 
        key set: 20337
        train:
        valid: 
        test: 

    v0.2: unknown / known split
    
"""

if __name__ == '__main__':

    # Dataset configuration 
    data_path = './data/drug/connectivity_top5bot5_pair.csv'
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
    dataset.set_mode('te')
    # dataset.shuffle_data()

    for idx, (k1, k1_l, k2, k2_l, sim) in enumerate(dataset.loader()):
        k1, k2, sim  = (np.array(xx) for xx in [k1, k2, sim])
        # dataset.decode_data(k1[0], k1_l[0], k2[0], k2_l[0], sim[0])
        pass

