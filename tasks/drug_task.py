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
        self.SR = [0.7, 0.1, 0.2] # split raio
        self.input_maxlen = 0

        # Static values
        self.FEATURE_NUM = 11
        self.ORG_INCHI_IDX = 6
        self.TAR_INCHI_IDX = 7
        self.KEY_LENGTH = 27
        self.SIM_IDX = 8
        self.NUM_SIM = 3

        # Key dictionaries (a part, b part, total)
        self.key2idx = {}
        self.idx2key = {}
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
        zipped = list(zip(p[0], p[1], p[2], p[3], p[4]))
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
    dataset.set_mode('te')
    # dataset.shuffle_data()

    for idx, (k1, k1_l, k2, k2_l, sim) in enumerate(dataset.loader()):
        k1, k2, sim  = (np.array(xx) for xx in [k1, k2, sim])
        dataset.decode_data(k1[0], k1_l[0], k2[0], k2_l[0], sim[0])
        pass

