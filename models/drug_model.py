import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import math
import sys
import logging

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


LOGGER = logging.getLogger(__name__)


class DrugModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, drug_embed_dim,
            lstm_layer, lstm_dropout, bi_lstm, linear_dropout, char_vocab_size,
            char_embed_dim, char_dropout, dist_fn, learning_rate,
            binary, is_mlp):

        super(DrugModel, self).__init__()

        # Save model configs
        self.drug_embed_dim = drug_embed_dim
        self.lstm_layer = lstm_layer
        self.char_dropout = char_dropout
        self.dist_fn = dist_fn
        self.binary = binary
        self.is_mlp = is_mlp

        # For rep_idx 0, 1
        if not is_mlp:
            self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, 
                                           padding_idx=0)
            self.lstm = nn.LSTM(char_embed_dim, drug_embed_dim, lstm_layer,
                                bidirectional=bi_lstm,
                                batch_first=True, dropout=lstm_dropout)
        # For rep_ix 2, 3
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, drug_embed_dim * 2)
            )

        self.dist_fc = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(drug_embed_dim * 2, 1)
        )

        # Get params and register optimizer
        info, params = self.get_model_params()
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # self.optimizer = optim.Adamax(params)
        if binary:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
        LOGGER.info(info)

    def init_lstm_h(self, batch_size):
        return (Variable(torch.zeros(
            	self.lstm_layer*2, batch_size, self.drug_embed_dim)).cuda(),
                Variable(torch.zeros(
            	self.lstm_layer*2, batch_size, self.drug_embed_dim)).cuda())

    # Set Siamese network as basic LSTM
    def siamese_sequence(self, inputs, length):

        # Character embedding
        c_embed = self.char_embed(inputs)
        # c_embed = F.dropout(c_embed, self.char_dropout)
        maxlen = inputs.size(1)

        if not self.training:
            # Sort c_embed
            _, sort_idx = torch.sort(length, dim=0, descending=True)
            _, unsort_idx = torch.sort(sort_idx, dim=0)
            maxlen = torch.max(length)

            # Pack padded sequence
            c_embed = c_embed.index_select(0, Variable(sort_idx).cuda())
            sorted_len = length.index_select(0, sort_idx).tolist()
            c_packed = pack_padded_sequence(c_embed, sorted_len, batch_first=True) 

        else:
            c_packed = c_embed


        # Run LSTM
        init_lstm_h = self.init_lstm_h(inputs.size(0))
        lstm_out, states = self.lstm(c_packed, init_lstm_h)

        hidden = torch.transpose(states[0], 0, 1).contiguous().view(
                                 -1, 2 * self.drug_embed_dim)
        if not self.training:
            # Unsort hidden states
            outputs = hidden.index_select(0, Variable(unsort_idx).cuda())
        else:
            outputs = hidden
        
        '''
        lstm_out = lstm_out.contiguous().view(-1, self.drug_embed_dim * 2)

        # Select length
        fw_lens = (torch.arange(0, inputs.size(0)).long()
                   * maxlen + length - 1).cuda()
        bw_lens = (torch.arange(0, inputs.size(0)).long() * maxlen).cuda()
        fw_selected = lstm_out[fw_lens,:]
        fw_states = fw_selected[:,:self.drug_embed_dim]
        bw_selected = lstm_out[bw_lens,:]
        bw_states = bw_selected[:,self.drug_embed_dim:]

        # Concat fw, bw states
        outputs = torch.cat([fw_states, bw_states], dim=1)
        '''

        return outputs
    
    def siamese_basic(self, inputs):
        return self.fc1(inputs.float())
    
    def distance_layer(self, vec1, vec2, distance='l1'):
	# Use Sigmoid for binary classification, otherwise tanh for regression
        if self.binary:
            nonl = F.sigmoid
        else:
            nonl = F.tanh

        if distance == 'cos':
            similarity = nonl(F.cosine_similarity(
                    vec1 + 1e-16, vec2 + 1e-16, dim=-1))
        elif distance == 'l1':
            similarity = nonl(self.dist_fc(torch.abs(vec1 - vec2)))
            similarity = similarity.squeeze(1)
        elif distance == 'l2':
            similarity = nonl(self.dist_fc(torch.abs((vec1 - vec2) ** 2)))
            similarity = similarity.squeeze(1)

        return similarity

    def forward(self, key1, key1_len, key2, key2_len):
        if not self.is_mlp:
            embed1 = self.siamese_sequence(key1, key1_len) 
            embed2 = self.siamese_sequence(key2, key2_len)
        else: 
            embed1 = self.siamese_basic(key1)
            embed2 = self.siamese_basic(key2)

        similarity = self.distance_layer(embed1, embed2, self.dist_fn)
        return similarity, embed1, embed2
    
    def get_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss

    def get_model_params(self):
        params = []
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())

        return '{}\nparam size: {:,}\n'.format(self, total_size), params

    def save_checkpoint(self, state, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        LOGGER.info('Save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        LOGGER.info('Load checkpoint %s' % filename)
        checkpoint = torch.load(filename)

        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

