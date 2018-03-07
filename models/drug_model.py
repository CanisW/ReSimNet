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
                                bidirectional=False,
                                batch_first=True, dropout=lstm_dropout)
        # For rep_ix 2, 3
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.Linear(hidden_dim, drug_embed_dim)
            )
            # Decoder is used for pretraining
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(drug_embed_dim, hidden_dim)
            )
            # self.init_layers()
        
        # Distance function
        self.dist_fc = nn.Linear(drug_embed_dim, 1)

        # Get params and register optimizer
        info, params = self.get_model_params()
        self.optimizer = optim.Adam(params, lr=learning_rate, weight_decay=0)
        # self.optimizer = optim.Adamax(params)
        if binary:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
        LOGGER.info(info)

    def init_lstm_h(self, batch_size):
        return (Variable(torch.zeros(
            	self.lstm_layer*1, batch_size, self.drug_embed_dim)).cuda(),
                Variable(torch.zeros(
            	self.lstm_layer*1, batch_size, self.drug_embed_dim)).cuda())

    def init_layers(self):
        nn.init.xavier_normal(self.encoder[0].weight.data)
        nn.init.xavier_normal(self.encoder[2].weight.data)
        nn.init.xavier_normal(self.encoder[4].weight.data)

    def pretrain_siamese(self, inputs, layer_num):
        inputs = inputs.float()
        
        # First layer
        hidden1 = self.encoder[1](self.encoder[0](inputs))
        recon1 = self.decoder[0](hidden1)
        if layer_num == 0:
            return self.criterion(recon1, inputs)

        # Second layer
        self.encoder[0].requires_grad = False
        hidden2 = self.encoder[3](self.encoder[2](hidden1))
        recon2 = self.decoder[1](hidden2)
        if layer_num == 1:
            hidden1 = hidden1.clone().detach()
            return self.criterion(recon2, hidden1)

        # Third layer
        self.encoder[2].requires_grad = False
        hidden3 = self.encoder[4](hidden2)
        recon3 = self.decoder[2](hidden3)
        if layer_num == 2:
            hidden2 = hidden2.clone().detach()
            return self.criterion(recon3, hidden2)

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
                                 -1, 1 * self.drug_embed_dim)
        if not self.training:
            # Unsort hidden states
            outputs = hidden.index_select(0, Variable(unsort_idx).cuda())
        else:
            outputs = hidden
        
        return outputs
    
    def siamese_basic(self, inputs):
        return self.encoder(inputs.float())
    
    def distance_layer(self, vec1, vec2, distance='l1'):
        if self.binary:
            nonl = F.sigmoid
        else:
            nonl = F.tanh

        if distance == 'cos':
            similarity = nonl(F.cosine_similarity(
                    vec1 + 1e-16, vec2 + 1e-16, dim=-1))
        elif distance == 'l1':
            similarity = nonl(torch.sum(vec1 - vec2, dim=1))
            # similarity = nonl(torch.sum(torch.abs(vec1 - vec2), dim=1))
        elif distance == 'l2':
            similarity = nonl(torch.sum(vec1 - vec2, dim=1))
            # similarity = nonl(torch.sum(torch.abs((vec1 - vec2) ** 2), dim=1))

        return similarity

    def forward(self, key1, key1_len, key2, key2_len, layer_num):
        if layer_num is not None:
            pretrain_loss = (self.pretrain_siamese(key1, layer_num) + 
                             self.pretrain_siamese(key2, layer_num))
        else:
            pretrain_loss = None
            self.encoder[0].requires_grad = True
            self.encoder[2].requires_grad = True
            self.encoder[4].requires_grad = True

        if not self.is_mlp:
            embed1 = self.siamese_sequence(key1, key1_len) 
            embed2 = self.siamese_sequence(key2, key2_len)
        else: 
            embed1 = self.siamese_basic(key1)
            embed2 = self.siamese_basic(key2)

        similarity = self.distance_layer(embed1, embed2, self.dist_fn)
        return similarity, embed1, embed2, pretrain_loss
    
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

