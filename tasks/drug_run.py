import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from torch.autograd import Variable

from models.root.utils import *


# Register each key embedding
def register_key(key, key_len, idx2char, embed, dictionary):
    key = key.data.tolist()
    key_len = key_len.tolist()
    for k, kl, emb in zip(key, key_len, embed):
        chars = ''.join(list(map(lambda x: idx2char[x], k[:kl])))
        if chars in dictionary:
            assert dictionary[chars] == emb.data.tolist()
        dictionary[chars] = emb.data.tolist()


# run a single epoch
def run_drug(model, dataset, args, train=False):
    total_step = 0.0
    total_metrics = np.zeros(2)
    tar_set = []
    pred_set = []
    key2vec = {}
    start_time = datetime.now()
    dataset.shuffle()

    for k1, k1_l, k2, k2_l, sim in dataset.loader(
                                            args.batch_size, args.sim_idx):
        k1, k1_l, k2, k2_l, sim  = (np.array(xx) for xx 
                                            in [k1, k1_l, k2, k2_l, sim])

        k1 = Variable(torch.LongTensor(k1)).cuda()
        k2 = Variable(torch.LongTensor(k2)).cuda()
        k1_l = torch.LongTensor(k1_l)
        k2_l = torch.LongTensor(k2_l)
        sim = Variable(torch.FloatTensor(sim)).cuda()

        model.optimizer.zero_grad()
        if train: model.train()
        else: model.eval()

        outputs, embed1, embed2 = model(k1, k1_l, k2, k2_l)
        if args.save_embed:
            register_key(k1, k1_l, dataset.idx2char, embed1, key2vec)
            register_key(k2, k2_l, dataset.idx2char, embed2, key2vec)
        loss = model.get_loss(outputs, sim)
        total_metrics[0] += loss.data[0]
        total_step += 1.0

        # Calculate corref
        tar_set += list(sim.data.cpu().numpy())
        pred_set += list(outputs.data.cpu().numpy())
        corref = np.corrcoef(tar_set, pred_set)[0][1]
        total_metrics[1] += corref

        if train and not args.save_embed:
            loss.backward()
            # nn.utils.clip_grad_norm(model.get_model_params(), 
            #         args.grad_max_norm)
            for p in model.get_model_params()[1]:
                if p.grad is not None:
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            model.optimizer.step()
        
        # Print for print step or at last
        if total_step % args.print_step == 0:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(
                    (total_step - 1) * args.batch_size + len(k1), 
                    dataset.dataset_len)
            _progress += ('{} '.format(int(total_step)) + ' iter '
                    + ' [{:.3f}, {:.3f}]'.format(loss.data[0], corref)
                    + ' time: {:2d}:{:2d}:{:2d}'.format(
                        et//3600, et%3600//60, et%60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

    # End of an epoch
    et = (datetime.now() - start_time).total_seconds()
    corref = np.corrcoef(tar_set, pred_set)[0][1]
    print('\n\ttotal metrics:\t' + str([float('{:.3f}'.format(tm))
        for tm in total_metrics/total_step]))
    print('\tpearson correlation: {:.3f}\t'.format(corref))

    # Save embed as pickle
    if args.save_embed:
        pickle.dump(key2vec, open('{}embed_{}.pkl'.format(
                    args.checkpoint_dir, args.model_name), 'wb'))

    return corref

