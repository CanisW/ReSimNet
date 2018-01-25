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
def register_key(key, embed, dictionary):
    assert len(key) == len(embed)
    for k, emb in zip(key, embed):
        if k in dictionary:
            # assert dictionary[chars] == emb.data.tolist()
            if dictionary[k] != emb.data.tolist():
                print('diff', sum(np.array(dictionary[k])) \
                        - sum(np.array(emb.data.tolist())))
            continue
        dictionary[k] = emb.data.tolist()


# run a single epoch
def run_drug(model, dataset, args, key2vec, train=False):
    total_step = 0.0
    total_metrics = [[],[],[],[],[]]
    tar_set = []
    pred_set = []
    kk_tar_set = []
    kk_pred_set = []
    ku_tar_set = []
    ku_pred_set = []
    uu_tar_set = []
    uu_pred_set = []
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

        # Binarize similarity
        if args.binary:
            sim = np.array([float(s > 0) for s in sim])
        else:
            if 'cscore' in args.data_path:
                sim = np.array([float(s / 100) for s in sim])

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
            register_key(key1, embed1, key2vec)
            register_key(key2, embed2, key2vec)
        loss = model.get_loss(outputs, sim)
        total_metrics[0] += [loss.data[0]]
        total_step += 1.0
        d_idx = (total_step - 1) * args.batch_size + k1.size(0)

        # Calculate corref
        tmp_tar = sim.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()

        # Metrics are different for regression and binary
        if not args.binary:
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
        else:
            tmp_pred = np.array([float(p > 0.5) for p in tmp_pred[:]])
            tar_set = tmp_tar[:]
            pred_set = tmp_pred[:]
            kk_tar_set = tmp_tar[kk_idx]
            kk_pred_set = tmp_pred[kk_idx]
            ku_tar_set = tmp_tar[ku_idx]
            ku_pred_set = tmp_pred[ku_idx]
            uu_tar_set = tmp_tar[uu_idx]
            uu_pred_set = tmp_pred[uu_idx]
        
            corref = sum(tar_set==pred_set)/len(tar_set)
            corref_kk = sum(kk_tar_set==kk_pred_set)/(len(kk_tar_set) + 1e-16)
            corref_ku = sum(ku_tar_set==ku_pred_set)/(len(ku_tar_set) + 1e-16)
            corref_uu = sum(uu_tar_set==uu_pred_set)/(len(uu_tar_set) + 1e-16)

        total_metrics[1] += [corref]
        if len(kk_idx) != 0:
            total_metrics[2] += [corref_kk]
        if len(ku_idx) != 0:
            total_metrics[3] += [corref_ku]
        if len(uu_idx) != 0:
            total_metrics[4] += [corref_uu]

        # Optimize model
        if train and not args.save_embed:
            loss.backward()
            nn.utils.clip_grad_norm(model.get_model_params()[1], 
                    args.grad_max_norm)
            # for p in model.get_model_params()[1]:
            #     if p.grad is not None:
            #         p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            model.optimizer.step()
        
        # Print for print step or at last
        if total_step % args.print_step == 0 or d_idx == dataset.length:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(d_idx, dataset.length)
            _progress += ('{} '.format(int(total_step)) + 'iter '
                    + 'Loss/Total/KK/KU/UU '
                    + str([float('{:.3f}'.format(sum(tm)/(len(tm) + 1e-16)))
                        for tm in total_metrics])
                    + ' time: {:2d}:{:2d}:{:2d}'.format(
                        et//3600, et%3600//60, et%60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

    # End of an epoch
    et = (datetime.now() - start_time).total_seconds()
    print('\n\ttotal metrics:\t' + str([float('{:.3f}'.format(
        sum(tm)/(len(tm) + 1e-16))) for tm in total_metrics]))

    if not args.binary:
        print('\tpearson correlation: {:.3f}\t'.format(corref))
        print('\tKK, KU, UU correlation: {:.3f}/{:.3f}/{:.3f}\t'.format(
              corref_kk, corref_ku, corref_uu))

    # Save embed as pickle
    if args.save_embed:
        pickle.dump(key2vec, open('{}embed_{}.pkl'.format(
                    args.checkpoint_dir, args.model_name), 'wb'), protocol=2)
        print('{} number of unique keys saved.'.format(len(key2vec)))

    if not args.binary:
        return corref
    else:
        return sum(total_metrics[1]) / len(total_metrics[1])

