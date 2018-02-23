import numpy as np
import sys
import copy
import pickle
import string
import os
import random
import csv
import torch

from os.path import expanduser
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class DrugDataset(object):
    def __init__(self, drug_id_path, drug_sub_path, drug_pair_path):

        self.initial_setting()
        # Build drug dictionary for id + sub
        self.drugs = self.process_drug_id(drug_id_path)
        self.append_drug_sub(drug_sub_path, self.drugs)

        # Save drug pair scores
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
        self.UNK = 'UNK'

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
        INCHIKEY_IDX = -1
        drugs = {}
        self.register_ichar(self.PAD)
        self.register_ichar(self.UNK)
        self.register_schar(self.PAD)
        self.register_schar(self.UNK)

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

            # Append drug sub id
            for drug, rep in drug2rep.items():
                if drug not in drugs:
                    drugs[drug] = [rep]
                else:
                    drugs[drug].append(rep)
            self.sub_lens.append(len(rep))

        print('Drug rep size {}\n'.format(self.sub_lens))

    def process_drug_pair(self, path):
        print('### Dug pair processing {}'.format(path))
        pair_scores = []
        REG_IDX = 5
        BI_IDX = -1

        with open(path) as f:
            csv_reader = csv.reader(f)
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    print(row)
                    print(row[REG_IDX], row[BI_IDX])
                    continue
                
                # Save drugs, score (real-valued), target (binary)
                drug1 = row[1]
                drug2 = row[2]
                reg_score = float(row[REG_IDX])
                bi_score = float(row[BI_IDX])
                assert drug1 in self.drugs and drug2 in self.drugs

                # Save each drug and scores
                pair_scores.append([drug1, drug2, [reg_score, bi_score]])

        print('Dataset size {}\n'.format(len(pair_scores)))
        return pair_scores
    
    def split_dataset(self, pair_scores, unk_test=True):
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
        random.shuffle(pair_scores)

        # Ready for train/valid/test
        train = []
        valid = []
        test = []
        valid_kk = valid_ku = valid_uu = 0
        test_kk = test_ku = test_uu = 0

        # If either one is unknown, add to test or valid
        for drug1, drug2, scores in pair_scores:
            if drug1 in self.unknown or drug2 in self.unknown:
                is_test = np.random.binomial(1, 
                                self.SR[2]/(self.SR[1]+self.SR[2]))

                if is_test:
                    test.append([drug1, drug2, scores])
                    if drug1 in self.unknown and drug2 in self.unknown:
                        test_uu += 1
                    else:
                        test_ku += 1
                else:
                    valid.append([drug1, drug2, scores])
                    if drug1 in self.unknown and drug2 in self.unknown:
                        valid_uu += 1
                    else:
                        valid_ku += 1

        # Fill known/known set with limit of split ratio
        for drug1, drug2, scores in pair_scores:
            if drug1 not in self.unknown and drug2 not in self.unknown:
                assert drug1 in self.known and drug2 in self.known

                if len(train) < len(pair_scores) * self.SR[0]:
                    train.append([drug1, drug2, scores])
                elif len(valid) < len(pair_scores) * self.SR[1]: 
                    valid.append([drug1, drug2, scores])
                    valid_kk += 1
                else:
                    test.append([drug1, drug2, scores])
                    test_kk += 1

        print('Train/Valid/Test split: {}/{}/{}'.format(
              len(train), len(valid), len(test)))
        print('Valid/Test KK,KU,UU: ({},{},{})/({},{},{})\n'.format(
              valid_kk, valid_ku, valid_uu, test_kk, test_ku, test_uu))

        return {'tr': train, 'va': valid, 'te': test}

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=5, s_idx=1):
        train_dataset = Representation(self.dataset['tr'], self.drugs, 
                                       self._rep_idx, s_idx=s_idx)
        train_sampler = SortedBatchSampler(train_dataset.lengths(),
                                           batch_size,
                                           shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        valid_dataset = Representation(self.dataset['va'], self.drugs, 
                                       self._rep_idx, s_idx=s_idx)
        valid_sampler = SortedBatchSampler(valid_dataset.lengths(),
                                           batch_size,
                                           shuffle=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        test_dataset = Representation(self.dataset['te'], self.drugs,
                                       self._rep_idx, s_idx=s_idx)
        test_sampler = SortedBatchSampler(test_dataset.lengths(),
                                           batch_size,
                                           shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        return train_loader, valid_loader, test_loader

    def collate_fn(self, batch):
        drug1_raws = [ex[0] for ex in batch]
        drug1_lens = torch.LongTensor([ex[2] for ex in batch])
        drug2_raws = [ex[3] for ex in batch]
        drug2_lens = torch.LongTensor([ex[5] for ex in batch])

        drug1_maxlen = max([len(ex[1]) for ex in batch])
        drug1_reps = torch.FloatTensor(len(batch), drug1_maxlen).zero_()
        drug2_maxlen = max([len(ex[4]) for ex in batch])
        drug2_reps = torch.FloatTensor(len(batch), drug2_maxlen).zero_()
        scores = torch.FloatTensor(len(batch)).zero_()

        for idx, ex in enumerate(batch):
            drug1_rep = ex[1]
            if self._rep_idx < 2:
                drug1_rep = list(map(lambda x: self.char2idx[x]
                                     if x in self.char2idx
                                     else self.char2idx[self.UNK], ex[1]))
            drug1_rep = torch.FloatTensor(drug1_rep)
            drug1_reps[idx, :drug1_rep.size(0)].copy_(drug1_rep)

            drug2_rep = ex[4]
            if self._rep_idx < 2:
                drug2_rep = list(map(lambda x: self.char2idx[x]
                                     if x in self.char2idx
                                     else self.char2idx[self.UNK], ex[4]))
            drug2_rep = torch.FloatTensor(drug2_rep)
            drug2_reps[idx, :drug2_rep.size(0)].copy_(drug2_rep)

            scores[idx] = ex[6]

        # Set to LongTensor if not mol2vec
        if self._rep_idx != 3:
            drug1_reps = drug1_reps.long()
            drug2_reps = drug2_reps.long()

        # Set as Variables
        drug1_reps = Variable(drug1_reps)
        drug2_reps = Variable(drug2_reps)
        scores = Variable(scores)
         
        return (drug1_raws, drug1_reps, drug1_lens, 
                drug2_raws, drug2_reps, drug2_lens, scores)

    def decode_data(self, d1, d1_l, d2, d2_l, score):
        d1 = d1.data.tolist()
        d2 = d2.data.tolist()
        if self._rep_idx >= 2:
            print('Drug1: {}, length: {}'.format(d1, d1_l))
            print('Drug2: {}, length: {}'.format(d2, d2_l))
        else:
            print('Drug1: {}, length: {}'.format(''.join(list(map(
                lambda x: self.idx2char[x], d1[:d1_l]))), d1_l))
            print('Drug2: {}, length: {}'.format(''.join(list(map(
                lambda x: self.idx2char[x], d2[:d2_l]))), d2_l))
        # print('Drug1: {}'.format(d1))
        # print('Drug2: {}'.format(d2))
        print('Score: {}\n'.format(score.data[0]))

    # rep_idx [0, 1, 2, 3]
    def set_rep(self, rep_idx):
        self._rep_idx = rep_idx
    
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
    def char_maxlen(self):
        if self._rep_idx == 0:
            return self.schar_maxlen
        elif self._rep_idx == 1:
            return self.ichar_maxlen
        else:
            return 0

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


class Representation(Dataset):
    def __init__(self, examples, drugs, rep_idx, s_idx):
        self.examples = examples
        self.drugs = drugs
        self.rep_idx = rep_idx
        self.s_idx = s_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        next_idx = index
        while (self.drugs[example[0]][self.rep_idx] == 'None' or
               self.drugs[example[1]][self.rep_idx] == 'None'):
            next_idx = (next_idx + 1) % len(self.examples)
            example = self.examples[next_idx]
        drug1, drug2, scores = example

        # Choose drug representation
        drug1_rep = self.drugs[drug1][self.rep_idx]
        drug1_len = len(drug1_rep)
        drug2_rep = self.drugs[drug2][self.rep_idx]
        drug2_len = len(drug2_rep)
        
        # Inchi None check
        if self.rep_idx == 1:
            assert drug1_rep != 'None' and drug2_rep != 'None'

        # s_idx == 1 means binary classification
        score = scores[self.s_idx]
        if self.s_idx == 1:
            score = float(score > 0)
        else:
            score = score / 100.
        return drug1, drug1_rep, drug1_len, drug2, drug2_rep, drug2_len, score
    
    def lengths(self):
        def get_longer_length(ex):
            drug1_len = len(self.drugs[ex[0]][self.rep_idx])
            drug2_len = len(self.drugs[ex[1]][self.rep_idx])
            length = drug1_len if drug1_len > drug2_len else drug2_len
            return [length, drug1_len, drug2_len]
        return [get_longer_length(ex) for ex in self.examples]


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(l1, l2, l3, np.random.random()) for l1, l2, l3 in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('l3', np.int_), 
                   ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])
    
    def __len__(self):
        return len(self.lengths)

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
    drug_id_path = './data/drug/drug_info_1.0_inchi.csv'
    drug_sub_path = ['./data/drug/drug_fingerprint_2.0_p2.pkl',
                     './data/drug/drug_mol2vec_2.0_p2.pkl']
    drug_pair_path = './data/drug/drug_cscore_pair_0.1.csv'
    save_preprocess = True
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
    dataset.set_rep(rep_idx=1)

    for idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(
            dataset.get_dataloader(batch_size=1600, s_idx=1)[1]):
        dataset.decode_data(d1_r[0], d1_l[0], d2_r[0], d2_l[0], score[0])
        pass

