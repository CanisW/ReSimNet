import os
import sys
import logging
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from torch.autograd import Variable

from tasks.drug_task import DrugDataset
from tasks.drug_run import run_drug, save_drug
from models.drug_model import DrugModel

from models.root.utils import *


LOGGER = logging.getLogger(__name__)


# Run settings
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, 
        default='./tasks/data/drug/drug(v0.3).pkl')
argparser.add_argument('--checkpoint_dir', type=str, default='./results/')
argparser.add_argument('--model_name', type=str, default='model.pth')
argparser.add_argument('--print_step', type=float, default=10)
argparser.add_argument('--train', type=int, default=1)
argparser.add_argument('--valid', type=int, default=1)
argparser.add_argument('--test', type=int, default=1)
argparser.add_argument('--resume', action='store_true', default=False)
argparser.add_argument('--debug', action='store_true', default=False)
argparser.add_argument('--save_embed', action='store_true', default=False)

# Train config
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--epoch', type=int, default=100)
argparser.add_argument('--learning_rate', type=float, default=1e-3)
argparser.add_argument('--weight_decay', type=float, default=0)
argparser.add_argument('--grad_max_norm', type=int, default=10)
argparser.add_argument('--grad_clip', type=int, default=10)

# Model config
argparser.add_argument('--binary', type=int, default=1)
argparser.add_argument('--hidden_dim', type=int, default=200)
argparser.add_argument('--drug_embed_dim', type=int, default=300)
argparser.add_argument('--lstm_layer', type=int, default=1)
argparser.add_argument('--lstm_dr', type=int, default=0.0)
argparser.add_argument('--linear_dr', type=int, default=0.0)
argparser.add_argument('--char_embed_dim', type=int, default=15)
argparser.add_argument('--s_idx', type=int, default=1)
argparser.add_argument('--rep_idx', type=int, default=0)
argparser.add_argument('--dist_fn', type=str, default='cos')
argparser.add_argument('--seed', type=int, default=1000)

args = argparser.parse_args()


# Create dirs
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


def run_experiment(model, dataset, run_fn, args):
    
    # Save embeddings and exit
    if args.save_embed:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        save_drug(model, dataset, args) 
        sys.exit()

    # Save and load model during experiments
    if args.train:
        if args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)

        best = 0.0
        for ep in range(args.epoch):
            print('- Training Epoch %d' % (ep+1))
            dataset.set_mode('tr', args.rep_idx)
            run_fn(model, dataset, args, train=True)

            if args.valid:
                print('- Validation')
                dataset.set_mode('va', args.rep_idx)
                curr = run_fn(model, dataset, args, train=False)
                if not args.resume and curr > best:
                    best = curr
                    model.save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()},
                        args.checkpoint_dir, args.model_name)
            print()
    
    if args.test:
        print('- Load Validation/Testing')
        if args.train or args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)
        dataset.set_mode('te', args.rep_idx)
        run_fn(model, dataset, args, train=False)
        save_drug(model, dataset, args) 
        print()


def get_dataset(path):
    return pickle.load(open(path, 'rb'))


def get_run_fn():
    return run_drug


def get_model(args, dataset):
    dataset.set_mode('invalid', args.rep_idx)
    model = DrugModel(input_dim=dataset.input_dim,
                      output_dim=1, 
                      hidden_dim=args.hidden_dim,
                      drug_embed_dim=args.drug_embed_dim,
                      lstm_layer=args.lstm_layer,
                      lstm_dropout=args.lstm_dr,
                      linear_dropout=args.linear_dr,
                      char_vocab_size=len(dataset.char2idx),
                      char_embed_dim=args.char_embed_dim,
                      dist_fn=args.dist_fn,
                      learning_rate=args.learning_rate,
                      binary=args.binary,
                      is_mlp=args.rep_idx > 1).cuda()
    return model


def init_logging():
    logging.basicConfig(
            format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
            level=logging.DEBUG)


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def main():
    # Initialize logging and prepare seed
    init_logging()
    LOGGER.info(args)
    init_seed(args.seed)

    # Get datset, run function, model
    dataset = get_dataset(args.data_path)
    run_fn = get_run_fn()
    model = get_model(args, dataset)

    # Run experiment
    run_experiment(model, dataset, run_fn, args)


if __name__ == '__main__':
    main()
