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
    def __init__(self, drug_id_path, drug_sub_path, drug_pair_path):
        self.initial_setting()
        # Build drug dictionary for id + sub
        self.drugs = self.process_drug_id(drug_id_path)
        self.append_drug_sub(drug_sub_path, self.drugs)

        self.pairs = self.process_drug_pair(drug_pair_path)
        self.dataset = self.split_dataset(self.pairs)

    def initial_setting(self):
        # Dataset split into train/valid/test
        self.drugs = {}
        self.pairs = []
        self.dataset = {'tr': [], 'va': [], 'te': []}
        self.SR = [0.7, 0.1, 0.2] # split ratio
        self.UR = 0.1 # Unknown ratio
        self.input_maxlen = 0

        # Drug dictionaries
        self.known = {}
        self.unknown = {}

        # Character dictionaries (smiles/inchikey chars)
        self.schar2idx = {}
        self.idx2schar = {}
        self.ichar2idx = {}
        self.idx2ichar = {}
        self.schar_maxlen = 0
        self.ichar_maxlen = 0
        self.sub_lens = []
        self.PAD = 'PAD'

    def register_schar(self, char):
        if char not in self.schar2idx:
            self.schar2idx[char] = len(self.schar2idx)
            self.idx2schar[len(self.idx2schar)] = char

    def register_ichar(self, char):
        if char not in self.ichar2idx:
            self.ichar2idx[char] = len(self.ichar2idx)
            self.idx2ichar[len(self.idx2ichar)] = char

    def process_drug_id(self, path):
        print('### Drug ID processing {}'.format(path))
        PERT_IDX = 0
        SMILES_IDX = 3
        INCHIKEY_IDX = 4
        drugs = {}
        self.register_ichar(self.PAD)
        self.register_schar(self.PAD)

        with open(path) as f:
            csv_reader = csv.reader(f)
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    continue

                # Add to drug dictionary
                drug = row[PERT_IDX]
                smiles = row[SMILES_IDX]
                inchikey = row[INCHIKEY_IDX]
                drugs[drug] = [smiles, inchikey] 
                
                # Update drug characters
                list(map(lambda x: self.register_schar(x), smiles))
                list(map(lambda x: self.register_ichar(x), inchikey))

                # Update max length
                self.schar_maxlen = self.schar_maxlen \
                        if self.schar_maxlen > len(smiles) else len(smiles)
                self.ichar_maxlen = self.ichar_maxlen \
                        if self.ichar_maxlen > len(inchikey) else len(inchikey)
        
        print('Drug dictionary size {}'.format(len(drugs)))
        print('Smiles char size {}'.format(len(self.schar2idx)))
        print('Smiles maxlen {}'.format(self.schar_maxlen))
        print('Inchikey char size {}'.format(len(self.ichar2idx)))
        print('Inchikey maxlen {}\n'.format(self.ichar_maxlen))
        return drugs
    
    def append_drug_sub(self, paths, drugs):
        for path in paths:
            print('### Drug subID appending {}'.format(path))
            drug2rep = pickle.load(open(path, 'rb'))
            assert len(drug2rep) == len(drugs)

            # Append drug sub id
            for drug, rep in drug2rep.items():
                drugs[drug].append(rep)
            self.sub_lens.append(len(rep))

        print('Drug rep size {}\n'.format(self.sub_lens))

    def process_drug_pair(self, path):
        print('### Dug pair processing {}'.format(path))
        drug1_set = []
        drug2_set = []
        scores = []

        with open(path) as f:
            csv_reader = csv.reader(f)
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    continue
                
                # Skip invalid rows, keys
                drug1 = row[1]
                drug2 = row[2]
                score = float(row[3])
                target = float(row[4])
                assert drug1 in self.drugs and drug2 in self.drugs

                # Save each drug and scores
                drug1_set.append(drug1)
                drug2_set.append(drug2)
                scores.append([score, target])

        pairs = [drug1_set, drug2_set, scores]

        print('Dataset size {}\n'.format(len(drug1_set)))
        return pairs
    
    def split_dataset(self, p, unk_test=True):
        print('### Split dataset')

        # Shuffle drugs dicitonary and split
        items = list(self.drugs.items())
        random.shuffle(items)
        if unk_test:
            self.known = dict(items[:int(-len(items) * self.UR)])
            self.unknown = dict(items[int(-len(items) * self.UR):])
        else:
            self.known = dict(items[:])
            self.unknown = dict()

        # Unknown check
        for unk, _ in self.unknown.items():
            assert unk not in self.known

        # Shuffle dataset
        zipped = list(zip(p[0], p[1], p[2]))
        random.shuffle(zipped)
        sf_p = list(zip(*zipped))

        # Ready for train/valid/test
        train = [[], [], []]
        valid = [[], [], []]
        test = [[], [], []]
        valid_kk = 0
        valid_ku = 0
        valid_uu = 0
        test_kk = 0
        test_ku = 0
        test_uu = 0

        # If either one is unknown, add to test or valid
        for drug1, drug2, scores in zip(sf_p[0], sf_p[1], sf_p[2]):
            if drug1 in self.unknown or drug2 in self.unknown:
                is_test = np.random.binomial(1, 
                                self.SR[2]/(self.SR[1]+self.SR[2]))

                if is_test:
                    test[0].append(drug1)
                    test[1].append(drug2)
                    test[2].append(scores)
                    if drug1 in self.unknown and drug2 in self.unknown:
                        test_uu += 1
                    else:
                        test_ku += 1
                else:
                    valid[0].append(drug1)
                    valid[1].append(drug2)
                    valid[2].append(scores)
                    if drug1 in self.unknown and drug2 in self.unknown:
                        valid_uu += 1
                    else:
                        valid_ku += 1

        # Fill known/known set with limit of split ratio
        for drug1, drug2, scores in zip(sf_p[0], sf_p[1], sf_p[2]):
            if drug1 not in self.unknown and drug2 not in self.unknown:
                assert drug1 in self.known and drug2 in self.known

                if len(train[0]) < len(sf_p[0]) * self.SR[0]:
                    train[0].append(drug1)
                    train[1].append(drug2)
                    train[2].append(scores)
                elif len(valid[0]) < len(sf_p[0]) * self.SR[1]: 
                    valid[0].append(drug1)
                    valid[1].append(drug2)
                    valid[2].append(scores)
                    valid_kk += 1
                else:
                    test[0].append(drug1)
                    test[1].append(drug2)
                    test[2].append(scores)
                    test_kk += 1

        print('Train/Valid/Test split: {}/{}/{}'.format(
              len(train[0]), len(valid[0]), len(test[0])))
        print('Valid/Test KK,KU,UU: ({},{},{})/({},{},{})\n'.format(
              valid_kk, valid_ku, valid_uu, test_kk, test_ku, test_uu))

        return {'tr': train, 'va': valid, 'te': test}
    
    def pad_drug(self, drug, maxlen, pad):
        while len(drug) != maxlen:
            drug.append(pad)
        assert len(drug) == maxlen
        return drug

    def loader(self, batch_size=16, s_idx=1):
        b_drug1 = []
        b_drug1_rep = []
        b_drug1_len = []
        b_drug2 = []
        b_drug2_rep = []
        b_drug2_len = []
        b_score = []
        d = self.dataset[self._mode]
        rep_idx = self._rep_idx

        for drug1, drug2, scores in zip(d[0], d[1], d[2]):
            drug1_rep = self.drugs[drug1][rep_idx]
            drug2_rep = self.drugs[drug2][rep_idx]
            drug1_len = 0
            drug2_len = 0

            # Smiles
            if rep_idx == 0:
                drug1_rep = list(map(lambda x: self.schar2idx[x], drug1_rep))
                drug1_len = len(drug1_rep)
                drug1_rep = self.pad_drug(drug1_rep, self.schar_maxlen, 
                                          self.schar2idx[self.PAD])
                drug2_rep = list(map(lambda x: self.schar2idx[x], drug2_rep))
                drug2_len = len(drug2_rep)
                drug2_rep = self.pad_drug(drug2_rep, self.schar_maxlen, 
                                          self.schar2idx[self.PAD])
            # Inchikey
            elif rep_idx == 1:
                drug1_rep = list(map(lambda x: self.ichar2idx[x], drug1_rep))
                drug1_len = len(drug1_rep)
                drug1_rep = self.pad_drug(drug1_rep, self.ichar_maxlen, 
                                          self.ichar2idx[self.PAD])
                drug2_rep = list(map(lambda x: self.ichar2idx[x], drug2_rep))
                drug2_len = len(drug2_rep)
                drug2_rep = self.pad_drug(drug2_rep, self.ichar_maxlen, 
                                          self.ichar2idx[self.PAD])
            # Fingerprint/Mol2vec (2, 3)
            b_drug1.append(drug1)
            b_drug1_rep.append(drug1_rep)
            b_drug1_len.append(drug1_len)
            b_drug2.append(drug2)
            b_drug2_rep.append(drug2_rep)
            b_drug2_len.append(drug2_len)
            b_score.append(scores[s_idx])

            if len(b_drug1) == batch_size:
                yield (b_drug1, b_drug1_rep, b_drug1_len, 
                       b_drug2, b_drug2_rep, b_drug2_len, b_score)
                del (b_drug1[:], b_drug1_rep[:], b_drug1_len[:],
                     b_drug2[:], b_drug2_rep[:], b_drug2_len[:], b_score[:])

        yield (b_drug1, b_drug1_rep, b_drug1_len, 
               b_drug2, b_drug2_rep, b_drug2_len, b_score)

    def shuffle(self):
        d = self.dataset[self._mode]
        zipped = list(zip(d[0], d[1], d[2]))
        random.shuffle(zipped)
        d[0], d[1], d[2] = zip(*zipped)

    def decode_data(self, d1, d1_l, d2, d2_l, score):
        print('Drug1: {}, length: {}'.format(''.join(list(map(
            lambda x: self.idx2char[x], d1[:d1_l]))), d1_l))
        print('Drug2: {}, length: {}'.format(''.join(list(map(
            lambda x: self.idx2char[x], d2[:d2_l]))), d2_l))
        print('Drug1: {}'.format(d1))
        print('Drug2: {}'.format(d2))
        print('Score: {}\n'.format(score))

    # Dataset mode ['tr', 'va'], rep_idx [0, 1, 2, 3]
    def set_mode(self, mode, rep_idx):
        self._mode = mode
        self._rep_idx = rep_idx

    @property
    def length(self):
        return len(self.dataset[self._mode][0])
    
    @property
    def char2idx(self):
        if self._rep_idx == 0:
            return self.schar2idx
        elif self._rep_idx == 1:
            return self.ichar2idx
        else:
            return {}  

    @property
    def idx2char(self):
        if self._rep_idx == 0:
            return self.idx2schar
        elif self._rep_idx == 1:
            return self.idx2ichar
        else:
            return {}

    @property
    def input_dim(self):
        if self._rep_idx == 0:
            return len(self.idx2schar)
        elif self._rep_idx == 1:
            return len(self.idx2ichar)
        elif self._rep_idx == 2:
            return 2048
        elif self._rep_idx == 3:
            return 300
        else:
            assert False, 'Wrong rep_idx {}'.format(rep_idx)

"""
[Version Note]
    v0.1: basic implementation
        key A, key B, char: 9165/5677/27 
        key set: 20337
        train:
        valid: 
        test: 
    v0.2: unknown / known split
    v0.3: append sub ids
    

drug_info_1.0.csv
- (drug_id, smiles, inchikey, target)

drug_cscore_pair_top1%bottom1%.csv
- (drug_id1, drug_id2, score, class)

drug_fingerprint_1.0_p3.pkl
- (drug_id, fingerprint)

drug_mol2vec_1.0_p3.pkl
- (drug_id, mol2vec)

"""


if __name__ == '__main__':

    # Dataset configuration 
    drug_id_path = './data/drug/drug_info_1.0.csv'
    drug_sub_path = ['./data/drug/drug_fingerprint_1.0_p3.pkl',
                     './data/drug/drug_mol2vec_1.0_p3.pkl']
    drug_pair_path = './data/drug/drug_cscore_pair_top1%bottom1%.csv'
    save_preprocess = False
    save_path = './data/drug/drug(tmp).pkl'
    load_path = './data/drug/drug(v0.3).pkl'

    # Save or load dataset
    if save_preprocess:
        dataset = DrugDataset(drug_id_path, drug_sub_path, drug_pair_path)
        pickle.dump(dataset, open(save_path, 'wb'))
        print('## Save preprocess %s' % save_path)
    else:
        print('## Load preprocess %s' % load_path)
        dataset = pickle.load(open(load_path, 'rb'))
   
    # Loader testing
    dataset.set_mode('te', rep_idx=1)
    # dataset.shuffle_data()

    for idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(
                                            dataset.loader(batch_size=1600)):
        d1, d2, score  = (np.array(xx) for xx in [d1, d2, score])
        print(idx)
        dataset.decode_data(d1_r[0], d1_l[0], d2_r[0], d2_l[0], score[0])
        pass

