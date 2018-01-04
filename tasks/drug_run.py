import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from torch.autograd import Variable

from models.root.utils import *


# run a single epoch
def run_drug(model, dataset, args, train=False):
    total_step = 0.0
    total_metrics = np.zeros(1)
    start_time = datetime.now()

    for k1, k1a, k1b, k2, k2a, k2b, sim in dataset.loader(args.batch_size):
        k1, k1a, k1b, k2, k2a, k2b, sim  = (np.array(xx) for xx in [k1, k1a, k1b,
                                            k2, k2a, k2b, sim])

        k1 = Variable(torch.LongTensor(k1)).cuda()
        k2 = Variable(torch.LongTensor(k2)).cuda()
        sim = Variable(torch.FloatTensor(sim)).cuda()

        model.optimizer.zero_grad()
        if train: model.train()
        else: model.eval()

        outputs = model(k1, k2)
        loss = model.get_loss(outputs, sim)
        total_metrics[0] += loss.data[0]
        total_step += 1.0

        if train:
            loss.backward()
            # nn.utils.clip_grad_norm(model.get_model_params(), 
            #         args.grad_max_norm)
            for p in model.get_model_params()[1]:
                if p.grad is not None:
                    p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            model.optimizer.step()
        
        # print step
        if total_step % args.print_step == 0:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(
                    (total_step - 1) * args.batch_size + len(k1), 
                    dataset.dataset_len)
            _progress += ('{} '.format(int(total_step)) + ' iter '
                    + ' [{:.3f}]'.format(loss.data[0])
                    + ' time: {:2d}:{:2d}:{:2d}'.format(
                        et//3600, et%3600//60, et%60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

    # end of an epoch
    et = (datetime.now() - start_time).total_seconds()
    print('\n\ttotal metrics:\t' + str([float('{:.3f}'.format(tm))
        for tm in total_metrics/total_step]))

    return total_metrics / total_step

