import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

from datetime import datetime
from torch.autograd import Variable
from models.root.utils import *


LOGGER = logging.getLogger(__name__)


# Run a single epoch
def run_drug(model, loader, dataset, args, train=False):
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

    for d_idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(loader):

        # Split for KK/KU/UU sets
        kk_idx = np.argwhere([a in dataset.known and b in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        ku_idx = np.argwhere([(a in dataset.unknown) != (b in dataset.unknown)
                              for a, b in zip(d1, d2)]).flatten()
        uu_idx = np.argwhere([a in dataset.unknown and b in dataset.unknown
                              for a, b in zip(d1, d2)]).flatten()
        assert len(kk_idx) + len(ku_idx) + len(uu_idx) == len(d1)

        # Grad zero + mode change
        model.optimizer.zero_grad()
        if train: model.train(train)
        else: model.eval()

        # Get outputs
        outputs, embed1, embed2 = model(d1_r.cuda(), d1_l, d2_r.cuda(), d2_l)
        loss = model.get_loss(outputs, score.cuda())
        total_metrics[0] += [loss.data[0]]
        total_step += 1.0

        # Calculate acc 
        tmp_tar = score.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()
        # print(tmp_tar[:10], tmp_pred[:10])

        # Metrics for binary classification
        tmp_pred = np.array([float(p > 0.5) for p in tmp_pred[:]])
        tar_set = tmp_tar[:]
        pred_set = tmp_pred[:]
        kk_tar_set = tmp_tar[kk_idx]
        kk_pred_set = tmp_pred[kk_idx]
        ku_tar_set = tmp_tar[ku_idx]
        ku_pred_set = tmp_pred[ku_idx]
        uu_tar_set = tmp_tar[uu_idx]
        uu_pred_set = tmp_pred[uu_idx]
    
        acc = sum(tar_set==pred_set)/len(tar_set)
        acc_kk = sum(kk_tar_set==kk_pred_set)/(len(kk_tar_set) + 1e-16)
        acc_ku = sum(ku_tar_set==ku_pred_set)/(len(ku_tar_set) + 1e-16)
        acc_uu = sum(uu_tar_set==uu_pred_set)/(len(uu_tar_set) + 1e-16)

        total_metrics[1] += [acc]
        if len(kk_idx) != 0:
            total_metrics[2] += [acc_kk]
        if len(ku_idx) != 0:
            total_metrics[3] += [acc_ku]
        if len(uu_idx) != 0:
            total_metrics[4] += [acc_uu]

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
        if d_idx % args.print_step == 0 or d_idx == (len(loader) - 1):
            et = int((datetime.now() - start_time).total_seconds())
            _progress = (
                '{}/{} | Loss: {:.3f} | Total Loss: {:.3f} | '.format(
                d_idx + 1, len(loader),
                sum(total_metrics[0])/(d_idx + 1), 
                sum(total_metrics[1])/(d_idx + 1)) +
                'KK: {:.3f} KU: {:.3f} UU: {:.3f} | '.format(
                sum(total_metrics[2])/(d_idx + 1), 
                sum(total_metrics[3])/(d_idx + 1),
                sum(total_metrics[4])/(d_idx + 1)) +
                'Time: {:2d}:{:2d}:{:2d}'.format(
                et//3600, et%3600//60, et%60))
            LOGGER.info(_progress)

    # End of an epoch
    et = (datetime.now() - start_time).total_seconds()
    LOGGER.info('total metrics:\t' + '\t'.join(['{:.3f}'.format(
        sum(tm)/len(loader)) for tm in total_metrics]))

    return sum(total_metrics[1]) / len(loader)


def save_drug(model, dataset, args):
    model.eval()
    key2vec = {}
    unk_cnt = 0

    # Iterate drug dictionary
    for idx, (drug, reps) in enumerate(dataset.drugs.items()):
        d1_r = reps[args.rep_idx]
        d1_l = len(d1_r)

        # For string data (smiles/inchikey)
        if args.rep_idx == 0 or args.rep_idx == 1:
            d1_r = list(map(lambda x: dataset.char2idx[x], d1_r))
            d1_l = len(d1_r)
            d1_r = dataset.pad_drug(d1_r, dataset.char_maxlen, 
                                          dataset.char2idx[dataset.PAD])

        # Real valued for mol2vec
        if dataset._rep_idx != 3:
            d1_r = Variable(torch.LongTensor(d1_r)).cuda()
        else:
            d1_r = Variable(torch.FloatTensor(d1_r)).cuda()
        d1_l = torch.LongTensor(np.array([d1_l]))
        d1_r = d1_r.unsqueeze(0)
        d1_l = d1_l.unsqueeze(0)

        # Run model amd save embed
        _, embed1, embed2 = model(d1_r, d1_l, d1_r, d1_l)
        assert embed1.data.tolist() == embed2.data.tolist()
        key2vec[drug] = [embed1.squeeze().data.tolist(), drug in dataset.known]

        if drug not in dataset.known:
            # assert drug in dataset.unknown
            unk_cnt += 1

        # Print progress
        _progress = '{}/{} saving drug embeddings'.format(
            idx, len(dataset.drugs))
        LOGGER.info(_progress)

    # assert unk_cnt == len(dataset.unknown)

    # Save embed as pickle
    pickle.dump(key2vec, open('{}embed_{}_toxic.pkl'.format(
                args.checkpoint_dir, args.model_name), 'wb'), protocol=2)
    LOGGER.info('{} number of unique drugs saved.'.format(len(key2vec)))
