import numpy as np
import sys
import pprint
import copy
import pickle
import nltk
import string
import os
import random
from os.path import expanduser

nltk.download('punkt')


class bAbIDataset(object):
    def __init__(self, data_dir, word2vec_path, word_embed_dim,
            preprocess_maxlen):

        self.word2vec_path = word2vec_path
        self.word_embed_dim = word_embed_dim
        self.initial_setting()

        self.get_initial_word_dict(data_dir)
        self.get_pretrained_word(word2vec_path)
        self.dataset = self.process_data(data_dir, preprocess_maxlen)

        self.pad_word(self.dataset['tr'])
        self.pad_word(self.dataset['va'])
        self.pad_word(self.dataset['te'])

    def initial_setting(self):
        # Special tokens
        self.PAD = '<PAD>'
        self.ANS = '-'

        # Initial word dictionary for GloVe filtering
        self.initial_word_dict = {}
        self.dataset = {'tr': [], 'va': [], 'te': []}
        self.input_maxlen = 0

        # Word dictionaries
        self.word2idx = {}
        self.idx2word = {}
        self.idx2vec = []
        self.word2idx[self.PAD] = 0
        self.idx2word[0] = self.PAD
        self.idx2vec.append([0.0] * self.word_embed_dim) # PAD
        self.word2idx[self.ANS] = 1
        self.idx2word[1] = self.ANS
        self.idx2vec.append([0.0] * self.word_embed_dim) # ANS
    
    def register_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word

    def tokenize(self, words):
        tokens = nltk.word_tokenize(words)
        if '6B' in self.word2vec_path:
            tokens = list(map(lambda x: x.lower(), tokens))
        return tokens
    
    def get_initial_word_dict(self, dir):
        print('### Getting initial word dictionary %s' % dir)
        for subdir, _, files, in os.walk(dir):
            for file in sorted(files):
                with open(os.path.join(subdir, file)) as f:
                    for line_idx, line in enumerate(f):
                        line = line[:-1]

                        def update_initial_word_dict(input_tokens):
                            for x in input_tokens:
                                if x not in self.initial_word_dict:
                                    self.initial_word_dict[x] = 0
                                self.initial_word_dict[x] += 1

                        if '\t' in line: # question, answer
                            question, answer, _ = line.split('\t')
                            question = ' '.join(question.split(' ')[1:])
                            update_initial_word_dict(self.tokenize(question))
                            
                            answer = (answer.split(',') 
                                    if ',' in answer else [answer])
                            if '6B' in self.word2vec_path:
                                answer = [w.lower() for w in answer]
                            update_initial_word_dict(answer)

                        else: # story
                            story_line = ' '.join(line.split(' ')[1:])
                            update_initial_word_dict(self.tokenize(story_line))

        print('initial word dictionary size', len(self.initial_word_dict))
        # print(self.initial_word_dict)

    def get_pretrained_word(self, path):
        print('\n### loading pretrained %s' % path)
        word2vec = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                try:
                    line = f.readline()
                    if not line: break
                    l_split = line.split()
                    word = l_split[0]
                    vec = list(map(lambda x: float(x), l_split[1:]))
                    word2vec[word] = vec
                except ValueError as e:
                    print(e)
        
        unk_cnt = 0
        for word, word_cnt in sorted(self.initial_word_dict.items()):
            if word != self.PAD and word != self.ANS:
                assert word_cnt > 0
                if word in word2vec:
                    self.register_word(word)
                    self.idx2vec.append(word2vec[word])
                else:
                    unk_cnt += 1
            else:
                print('UNK, ANS token overlapping')

        # Debug pretrained vectors
        assert word2vec['apple'] == self.idx2vec[self.word2idx['apple']]
        assert len(self.word2idx) == len(self.idx2vec) == len(self.idx2word)
        print('Apple idx {}'.format(self.word2idx['apple']))
        print('Pretrained vectors {}, unknowns {}'.format(
            np.asarray(self.idx2vec).shape, unk_cnt))
        print('Word dictionary changes from {} to {}\n'.format(
            len(self.initial_word_dict), len(self.word2idx)))

    def process_data(self, dir, preprocess_maxlen):
        word_idxs = {'tr': [], 'va': [], 'te': []}
        anss = {'tr': [], 'va': [], 'te': []}
        ans_locs = {'tr': [], 'va': [], 'te': []}
        input_lens = {'tr': [], 'va': [], 'te': []}
        qa_nums = {'tr': [], 'va': [], 'te': []}

        for subdir, _, files, in os.walk(dir):
            for file in sorted(files):
                print('### Processing {}{}'.format(dir, file))
                with open(os.path.join(subdir, file)) as f:
                    # QA type => for logging
                    qa_num = file.split('_')[0][2:]

                    # Train/Valid/Test
                    set_type = file.split('_')[-1][:-4] 
                    story = []
                    answers = []
                    ans_loc = []

                    for line_idx, line in enumerate(f):
                        line = line[:-1]
                        story_idx = int(line.split(' ')[0])

                        # Start of the story
                        if story_idx == 1: 

                            # Process previous story
                            if len(story) > 0 and \
                                    len(story) <= preprocess_maxlen:
                                self.input_maxlen = (self.input_maxlen
                                        if self.input_maxlen > len(story)
                                        else len(story))
                                word_idxs[set_type[:2]].append(story)
                                anss[set_type[:2]].append(answers)
                                ans_locs[set_type[:2]].append(ans_loc)
                                input_lens[set_type[:2]].append(len(story))
                                qa_nums[set_type[:2]].append(qa_num)

                            story = []
                            answers = []
                            ans_loc = []

                        if '\t' in line: # question, answer
                            question, answer, _ = line.split('\t')
                            question = ' '.join(question.split(' ')[1:])
                            q_split = list(map(lambda x: self.word2idx[x], 
                                self.tokenize(question)))
                            story += q_split

                            answer = (answer.split(',') 
                                    if ',' in answer else [answer])
                            if '6B' in self.word2vec_path:
                                answer = [w.lower() for w in answer]
                            answer = list(map(
                                lambda x: self.word2idx[x], answer))
                            answers += [answer]
                            ans_loc += [list(range(len(story), 
                                len(story) + len(answer)))]
                            story += ([self.word2idx[self.ANS]] * len(answer))

                        else: # story
                            story_line = ' '.join(line.split(' ')[1:])
                            s_split = list(map(lambda x: self.word2idx[x],
                                self.tokenize(story_line)))
                            story += s_split
    
        dataset = {'tr': [word_idxs['tr'], anss['tr'], ans_locs['tr'],
                          input_lens['tr'], qa_nums['tr']],
                   'va': [word_idxs['va'], anss['va'], ans_locs['va'],
                          input_lens['va'], qa_nums['va']],
                   'te': [word_idxs['te'], anss['te'], ans_locs['te'],
                          input_lens['te'], qa_nums['te']]}

        print('Dataset size {}/{}/{}'.format(
            len(dataset['tr'][0]), len(dataset['va'][0]), len(dataset['te'][0])))
        print('Max input length', self.input_maxlen, end='\n\n')

        return dataset

    def pad_word(self, d):
        for datum_idx, w in enumerate(d[0]):
            def pad_max(x, max, pad):
                while len(x) < max:
                    x.append(pad)
                assert len(x) == max

            # Pad word level
            pad_max(w, self.input_maxlen, self.word2idx[self.PAD])

    def loader(self, batch_size=16, maxlen=None):
        if maxlen is None: 
            maxlen = float("inf")

        d = self.dataset[self._mode]
        batch_w, batch_a, batch_al, batch_l, batch_qn = ([] for _ in range(5))

        for idx in range(0, len(d[0])):
            w, a, al, l, qn = list(map(lambda x: x[idx], d))

            if l > maxlen:
                continue
            
            batch_w.append(w)
            batch_a.append(a)
            batch_al.append(al)
            batch_l.append(l)
            batch_qn.append(qn)

            if len(batch_w) == batch_size:
                yield (batch_w, batch_a, batch_al, batch_l, batch_qn)
                del (batch_w[:], batch_a[:], batch_al [:], batch_l[:], 
                        batch_qn[:])

        if len(batch_w) > 0:
            yield (batch_w, batch_a, batch_al, batch_l, batch_qn)

    def shuffle_data(self):
        d = self.dataset[self._mode]
        zipped = list(zip(d[0], d[1], d[2], d[3], d[4]))
        random.shuffle(zipped)
        d[0], d[1], d[2], d[3], d[4] = zip(*zipped)

    def decode_data(self, w, a, al, l, qn):
        # print('Word idxs: {}\n'.format(w[:l]))
        print('Story: {}\n'.format(' '.join(list(map(
            lambda x: self.idx2word[x], w[:l])))))
        print('Answer: {}, Location: {}'.format(list(map(
            lambda x: list(map(
            lambda y: self.idx2word[y], x)), a)), al))
        print('Length: {}, QA Num: {}\n'.format(l, qn))

    # Dataset mode ['tr', 'va']
    def set_mode(self, mode):
        self._mode = mode

    @property
    def dataset_len(self):
        return len(self.dataset[self._mode][0])


"""
[Version Note]
    v0.1: basic implementation + QA num + shuffle/decode
        word dict: 158 => 160
        word2vec: 6B.300d
        max inp: 1920
        train: 56376
        valid: 6245
        test: 6247
    
"""

if __name__ == '__main__':

    # Dataset configuration 
    data_dir = './data/babi/en/'
    word2vec_type = 6  # [6, 300]
    word_embed_dim = 300 # [50, 100, 200, 300]
    word2vec_path = expanduser('~') + (
            '/common/glove/glove.' + 
            str(word2vec_type) + 'B.' +
            str(word_embed_dim) +'d.txt' )
    preprocess_maxlen = float("inf")
    save_preprocess = False
    save_path = './data/babi/babi(tmp).pkl'
    load_path = './data/babi/babi(tmp).pkl'

    # Save or load dataset
    if save_preprocess:
        dataset = bAbIDataset(data_dir, word2vec_path, word_embed_dim,
                preprocess_maxlen)
        pickle.dump(dataset, open(save_path, 'wb'))
        print('## Save preprocess %s' % load_path)
    else:
        print('## Load preprocess %s' % load_path)
        dataset = pickle.load(open(load_path, 'rb'))
   
    # Loader testing
    dataset.set_mode('va')
    dataset.shuffle_data()
    maxlen = float("inf")

    for idx, (w, a, al, l, qn) in enumerate(
            dataset.loader(batch_size=300, maxlen=maxlen)):
        w, a, al, l, qn = [np.array(w), np.array(a), np.array(al), 
                np.array(l), np.array(qn)]
        # print(idx, w.shape, a.shape, l.shape, qn.shape)
        dataset.decode_data(w[0], a[0], al[0], l[0], qn[0])

