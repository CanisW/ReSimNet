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
    total_metrics = np.zeros(5)
    tar_set = []
    pred_set = []
    kk_tar_set = []
    kk_pred_set = []
    ku_tar_set = []
    ku_pred_set = []
    uu_tar_set = []
    uu_pred_set = []
    key2vec = {}
    start_time = datetime.now()
    dataset.shuffle()

    for k1, k1_l, k2, k2_l, sim in dataset.loader(
                                            args.batch_size, args.sim_idx):

        # Split for KK/KU/UU sets
        k1, k1_l, k2, k2_l, sim  = (np.array(xx) for xx 
                                            in [k1, k1_l, k2, k2_l, sim])
        key1 = [''.join(list(map(lambda x: dataset.idx2char[x], ks1[:ks1_l])))
                for ks1, ks1_l in zip(k1, k1_l)]
        key2 = [''.join(list(map(lambda x: dataset.idx2char[x], ks2[:ks2_l])))
                for ks2, ks2_l in zip(k2, k2_l)]
        kk_idx = np.argwhere([a in dataset.known and b in dataset.known
                              for a, b in zip(key1, key2)]).flatten()
        ku_idx = np.argwhere([(a in dataset.unknown) != (b in dataset.unknown)
                              for a, b in zip(key1, key2)]).flatten()
        uu_idx = np.argwhere([a in dataset.unknown and b in dataset.unknown
                              for a, b in zip(key1, key2)]).flatten()
        assert len(kk_idx) + len(ku_idx) + len(uu_idx) == len(k1)

        # Wrap as Tensor/Variable
        k1 = Variable(torch.LongTensor(k1)).cuda()
        k2 = Variable(torch.LongTensor(k2)).cuda()
        k1_l = torch.LongTensor(k1_l)
        k2_l = torch.LongTensor(k2_l)
        sim = Variable(torch.FloatTensor(sim)).cuda()

        # Grad zero + mode change
        model.optimizer.zero_grad()
        if train: model.train(train)
        else: model.eval()

        # Get outputs
        outputs, embed1, embed2 = model(k1, k1_l, k2, k2_l)
        if args.save_embed:
            register_key(k1, k1_l, dataset.idx2char, embed1, key2vec)
            register_key(k2, k2_l, dataset.idx2char, embed2, key2vec)
        loss = model.get_loss(outputs, sim)
        total_metrics[0] += loss.data[0]
        total_step += 1.0
        d_idx = (total_step - 1) * args.batch_size + k1.size(0)

        # Calculate corref
        tmp_tar = sim.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()
        tar_set += list(tmp_tar[:])
        pred_set += list(tmp_pred[:])
        kk_tar_set += list(tmp_tar[kk_idx])
        kk_pred_set += list(tmp_pred[kk_idx])
        ku_tar_set += list(tmp_tar[ku_idx])
        ku_pred_set += list(tmp_pred[ku_idx])
        uu_tar_set += list(tmp_tar[uu_idx])
        uu_pred_set += list(tmp_pred[uu_idx])
        corref = np.corrcoef(tar_set, pred_set)[0][1]
        corref_kk = np.corrcoef(kk_tar_set, kk_pred_set)[0][1]
        corref_ku = np.corrcoef(ku_tar_set, ku_pred_set)[0][1]
        corref_uu = np.corrcoef(uu_tar_set, uu_pred_set)[0][1]
        total_metrics[1] += corref
        total_metrics[2] += corref_kk
        total_metrics[3] += corref_ku
        total_metrics[4] += corref_uu

        # Optimize model
        if train and not args.save_embed:
            loss.backward()
            # nn.utils.clip_grad_norm(model.get_model_params(), 
            #         args.grad_max_norm)
            for p in model.get_model_params()[1]:
                if p.grad is not None:
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            model.optimizer.step()
        
        # Print for print step or at last
        if total_step % args.print_step == 0 or d_idx == dataset.length:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(d_idx, dataset.length)
            _progress += ('{} '.format(int(total_step)) + ' iter '
                    + ' [{:.3f}, cor:{:.3f}'.format(loss.data[0], corref)
                    + ', kk:{:.3f}, ku:{:.3f}, uu:{:.3f}]'.format(
                        corref_kk, corref_ku, corref_uu)
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
    print('\tKK, KU, UU correlation: {:.3f}/{:.3f}/{:.3f}\t'.format(
          corref_kk, corref_ku, corref_uu))

    # Save embed as pickle
    if args.save_embed:
        pickle.dump(key2vec, open('{}embed_{}.pkl'.format(
                    args.checkpoint_dir, args.model_name), 'wb'))

    return corref

