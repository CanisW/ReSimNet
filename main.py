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
from tasks.drug_run import *
from models.drug_model import DrugModel

from models.root.utils import *


LOGGER = logging.getLogger()

DATA_PATH = './tasks/data/drug/drug(v0.1).pkl'
DRUG_PATH = './tasks/data/drug/validation/sider_smiles_3.pkl'
CKPT_DIR = './results/'
MODEL_NAME = 'test.mdl'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')



# Run settings
argparser = argparse.ArgumentParser()
argparser.register('type', 'bool', str2bool)

argparser.add_argument('--data-path', type=str, default=DATA_PATH,
                       help='Dataset path')
argparser.add_argument('--drug-path', type=str, default=DRUG_PATH,
                       help='Input drug dictionary path')
argparser.add_argument('--checkpoint-dir', type=str, default=CKPT_DIR,
                       help='Directory for model checkpoint')
argparser.add_argument('--model-name', type=str, default=MODEL_NAME,
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=float, default=100,
                       help='Display steps')
argparser.add_argument('--train', type='bool', default=True,
                       help='Enable training')
argparser.add_argument('--valid', type='bool', default=True,
                       help='Enable validation')
argparser.add_argument('--test', type='bool', default=True,
                       help='Enable testing')
argparser.add_argument('--resume', type='bool', default=False,
                       help='Resume saved model')
argparser.add_argument('--debug', type='bool', default=False,
                       help='Run as debug mode')
argparser.add_argument('--save-embed', type='bool', default=False,
                       help='Save embeddings with loaded model')
argparser.add_argument('--save-prediction', type='bool', default=False,
                       help='Save predictions with loaded model')

# Train config
argparser.add_argument('--batch-size', type=int, default=32)
argparser.add_argument('--epoch', type=int, default=40)
argparser.add_argument('--learning-rate', type=float, default=1e-3)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--grad-max-norm', type=int, default=10)
argparser.add_argument('--grad-clip', type=int, default=10)

# Model config
argparser.add_argument('--binary', type='bool', default=True)
argparser.add_argument('--hidden-dim', type=int, default=200)
argparser.add_argument('--drug-embed-dim', type=int, default=300)
argparser.add_argument('--lstm-layer', type=int, default=1)
argparser.add_argument('--lstm-dr', type=float, default=0.0)
argparser.add_argument('--linear-dr', type=float, default=0.0)
argparser.add_argument('--char-embed-dim', type=int, default=20)
argparser.add_argument('--s-idx', type=int, default=1)
argparser.add_argument('--rep-idx', type=int, default=0)
argparser.add_argument('--dist-fn', type=str, default='l2')
argparser.add_argument('--seed', type=int, default=3)

args = argparser.parse_args()


# Create dirs
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


def run_experiment(model, dataset, run_fn, args):

    # Get dataloaders
    train_loader, valid_loader, test_loader = dataset.get_dataloader(
        batch_size=args.batch_size, s_idx=args.s_idx) 
    
    # Save embeddings and exit
    if args.save_embed:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, train=False)
        drugs = pickle.load(open(args.drug_path, 'rb'))
        save_drug(model, drugs, dataset, args) 
        sys.exit()
    
    # Save predictions on test dataset and exit
    if args.save_prediction:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, train=False)
        save_prediction(model, test_loader, dataset, args)
        sys.exit()

    # Save and load model during experiments
    if args.train:
        if args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)

        best = 0.0
        for ep in range(args.epoch):
            LOGGER.info('Training Epoch %d' % (ep+1))
            run_fn(model, train_loader, dataset, args, train=True)

            if args.valid:
                LOGGER.info('Validation')
                curr = run_fn(model, valid_loader, dataset, args, train=False)
                if not args.resume and curr > best:
                    best = curr
                    model.save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()},
                        args.checkpoint_dir, args.model_name)
    
    if args.test:
        LOGGER.info('Load Validation/Testing')
        if args.train or args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)
        run_fn(model, valid_loader, dataset, args, train=False)
        run_fn(model, test_loader, dataset, args, train=False)


def get_dataset(path):
    return pickle.load(open(path, 'rb'))


def get_run_fn(args):
    if args.binary:
        return run_binary
    else:
        return run_regression


def get_model(args, dataset):
    dataset.set_rep(args.rep_idx)
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


def init_logging(args):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    # For logfile writing
    logfile = logging.FileHandler(
        args.checkpoint_dir + args.model_name + '.txt', 'w')
    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def main():
    # Initialize logging and prepare seed
    init_logging(args)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))
    LOGGER.info(args)
    init_seed(args.seed)

    # Get datset, run function, model
    dataset = get_dataset(args.data_path)
    run_fn = get_run_fn(args)
    model = get_model(args, dataset)

    # Run experiment
    run_experiment(model, dataset, run_fn, args)


if __name__ == '__main__':
    main()
