import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import csv

from datetime import datetime
from sklearn.metrics import f1_score
from torch.autograd import Variable
from models.root.utils import *


LOGGER = logging.getLogger(__name__)


def run_binary(model, loader, dataset, args, train=False):
    total_step = 0.0
    metrics = {'loss':[]}
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
        metrics['loss'] += [loss.data[0]]
        total_step += 1.0

        # Metrics for binary classification
        tmp_tar = score.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()
        tmp_pred = np.array([float(p > 0.5) for p in tmp_pred[:]])

        # Accumulate for final evaluation
        tar_set += list(tmp_tar[:])
        pred_set += list(tmp_pred[:])
        kk_tar_set += list(tmp_tar[kk_idx])
        kk_pred_set += list(tmp_pred[kk_idx])
        ku_tar_set += list(tmp_tar[ku_idx])
        ku_pred_set += list(tmp_pred[ku_idx])
        uu_tar_set += list(tmp_tar[uu_idx])
        uu_pred_set += list(tmp_pred[uu_idx])
    
        # Calculate current f1 scores
        f1 = f1_score(list(tmp_tar[:]), list(tmp_pred[:]))
        f1_kk = f1_score(list(tmp_tar[kk_idx]), list(tmp_pred[kk_idx]))
        f1_ku = f1_score(list(tmp_tar[ku_idx]), list(tmp_pred[ku_idx]))
        f1_uu = f1_score(list(tmp_tar[uu_idx]), list(tmp_pred[uu_idx]))

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
                '{}/{} | Loss: {:.3f} | Total F1: {:.3f} | '.format(
                d_idx + 1, len(loader), loss.data[0], f1) +
                'KK: {:.3f} KU: {:.3f} UU: {:.3f} | '.format(
                f1_kk, f1_ku, f1_uu) +
                '{:2d}:{:2d}:{:2d}'.format(
                et//3600, et%3600//60, et%60))
            LOGGER.info(_progress)

    # Calculate acuumulated f1 scores
    f1 = f1_score(tar_set, pred_set, average='binary')
    f1_kk = f1_score(kk_tar_set, kk_pred_set, average='binary')
    f1_ku = f1_score(ku_tar_set, ku_pred_set, average='binary')
    f1_uu = f1_score(uu_tar_set, uu_pred_set, average='binary')

    # TODO add spearman correlation

    # End of an epoch
    et = (datetime.now() - start_time).total_seconds()
    LOGGER.info('Results (Loss/F1/KK/KU/UU): {:.3f}\t{:.3f}\t'.format(
        sum(metrics['loss'])/len(metrics['loss']), f1) +
        '{:.3f}\t{:.3f}\t{:.3f}'.format(
        f1_kk, f1_ku, f1_uu))

    return f1_ku


# TODO: change to regression (add pearson + spearman correlation)
def run_regression(model, loader, dataset, args, train=False):
    return None


# Outputs response embeddings for a given dictionary
def save_drug(model, dictionary, dataset, args):
    model.eval()
    key2vec = {}
    known_cnt = 0

    # Iterate drug dictionary
    for idx, (drug, rep) in enumerate(dictionary.items()):
        d1_r = rep[0]
        d1_k = rep[1]
        d1_l = len(d1_r)

        # For string data (smiles/inchikey)
        if args.rep_idx == 0 or args.rep_idx == 1:
            d1_r = list(map(lambda x: dataset.char2idx[x]
                        if x in dataset.char2idx
                        else dataset.char2idx[dataset.UNK], d1_r))
            d1_l = len(d1_r)

        # Real valued for mol2vec
        if args.rep_idx != 3:
            d1_r = Variable(torch.LongTensor(d1_r)).cuda()
        else:
            d1_r = Variable(torch.FloatTensor(d1_r)).cuda()
        d1_l = torch.LongTensor(np.array([d1_l]))
        d1_r = d1_r.unsqueeze(0)
        d1_l = d1_l.unsqueeze(0)

        # Run model amd save embed
        _, embed1, embed2 = model(d1_r, d1_l, d1_r, d1_l)
        assert embed1.data.tolist() == embed2.data.tolist()
        """
        known = False
        for pert_id, _ in dataset.drugs.items():
            if drug == pert_id:
                known = True
                known_cnt += 1
                break
        """
        key2vec[drug] = [embed1.squeeze().data.tolist(), d1_k]

        # Print progress
        if idx % args.print_step == 0 or idx == len(dictionary) - 1:
            _progress = '{}/{} saving drug embeddings..'.format(
                idx + 1, len(dictionary))
            LOGGER.info(_progress)

    # Save embed as pickle
    pickle.dump(key2vec, open('{}toxcast_embed_{}.pkl'.format(
                args.checkpoint_dir, args.model_name), 'wb'), protocol=2)
    LOGGER.info('{}/{} number of known drugs.'.format(known_cnt, len(key2vec)))


# Outputs pred vs label scores given a dataloader
def save_prediction(model, loader, dataset, args):
    model.eval()
    csv_writer = csv.writer(open(args.checkpoint_dir + 'pred_' + 
                                 args.model_name + '.csv', 'w'))
    csv_writer.writerow(['pert1', 'pert1_known', 'pert2', 'pert2_known',
                         'prediction', 'target'])

    for d_idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(loader):

        # Run model for getting predictions
        outputs, _, _ = model(d1_r.cuda(), d1_l, d2_r.cuda(), d2_l)
        predictions = outputs.data.cpu().numpy()
        targets = score.data.tolist()

        for a1, a2, a3, a4 in zip(d1, d2, predictions, targets):
            csv_writer.writerow([a1, a1 in dataset.known, 
                                 a2, a2 in dataset.known, a3, a4])

        # Print progress
        if d_idx % args.print_step == 0 or d_idx == len(loader) - 1:
            _progress = '{}/{} saving drug predictions..'.format(
                d_idx + 1, len(loader))
            LOGGER.info(_progress)


# Outputs pred scores for new pair dataset
def save_pair_scores(model, loader, dataset, args):
    model.eval()
    csv_writer = csv.writer(open(args.checkpoint_dir + 'scores_' + 
                                 args.model_name + '.csv', 'w'))
    csv_writer.writerow(['pert1', 'pert1_known', 'pert2', 'pert2_known',
                         'prediction'])

    for d_idx, (d1, d2) in enumerate(loader):

        # Run model for getting predictions
        outputs, _, _ = model(d1_r.cuda(), d1_l, d2_r.cuda(), d2_l)
        predictions = outputs.data.cpu().numpy()

        for a1, a2, a3 in zip(d1, d2, predictions):
            csv_writer.writerow([a1, a1 in dataset.known, 
                                 a2, a2 in dataset.known, a3])

        # Print progress
        if d_idx % args.print_step == 0 or d_idx == len(loader) - 1:
            _progress = '{}/{} saving new pair predictions..'.format(
                d_idx + 1, len(loader))
            LOGGER.info(_progress)
