import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import math
import sys

from torch.autograd import Variable


class DrugModel(nn.Module):
    def __init__(self, output_dim, lstm_dim, lstm_layer, 
            char_vocab_size, char_embed_dim, learning_rate):

        super(DrugModel, self).__init__()

        # Save model configs
        self.lstm_dim = lstm_dim
        self.lstm_layer = lstm_layer

        # Basic modules
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim)
        self.lstm = nn.LSTM(char_embed_dim, lstm_dim, lstm_layer,
                batch_first=True)

        # Get params and register optimizer
        info, params = self.get_model_params()
        # self.optimizer = optim.RMSprop(params, lr=learning_rate,
        #         alpha=0.95, momentum=0.9, eps=1e-10)
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.criterion = nn.MSELoss()
        print(info)

    def init_lstm_h(self, batch_size):
        return (Variable(torch.zeros(
            self.lstm_layer*1, batch_size, self.lstm_dim)).cuda(),
                Variable(torch.zeros(
            self.lstm_layer*1, batch_size, self.lstm_dim)).cuda())

    # Set Siamese network as basic LSTM
    def siamese_network(self, inputs):
        # Character embedding
        c_embed = self.char_embed(inputs)

        # Run LSTM
        init_lstm_h = self.init_lstm_h(inputs.size(0))
        lstm_out, _ = self.lstm(c_embed, init_lstm_h)
        return lstm_out[:,-1,:]
    
    # Calculate similarity score of vec1 and vec2
    def distance_layer(self, vec1, vec2, distance='cosine'):
        if distance == 'cosine':
            similarity = F.cosine_similarity(
                    vec1 + 1e-16, vec2 + 1e-16, dim=-1)
        elif distance == 'l1':
            similarity = torch.abs(vec1 - vec2)
        elif distance == 'l2':
            similarity = torch.abs((vec1 - vec2) ** 2)

        return similarity

    def forward(self, key1, key2):
        embed1 = self.siamese_network(key1) 
        embed2 = self.siamese_network(key2)
        similarity = self.distance_layer(embed1, embed2)

        return similarity
    
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
                # print(p.size())
        return '{}\nparam size: {:,}\n'.format(self, total_size), params

    def save_checkpoint(self, state, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        print('\t=> save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        print('\t=> load checkpoint %s' % filename)
        checkpoint = torch.load(filename)

        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

