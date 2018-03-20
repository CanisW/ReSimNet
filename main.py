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
from functools import partial
from torch.autograd import Variable

from tasks.drug_task import DrugDataset
from tasks.drug_run import *
from models.drug_model import DrugModel
from models.root.utils import *


LOGGER = logging.getLogger()

DATA_PATH = './tasks/data/drug/drug(v0.5).pkl'  # For training (Pair scores)
#DATA_PATH = './tasks/data/drug/drug(v0.1_graph).pkl' 
DRUG_DIR = './tasks/data/drug/validation/'      # For validation (ex: tox21)
#DRUG_FILES = ['BBBP_fingerprint_3.pkl',
#              'clintox_fingerprint_3.pkl',
#              'sider_fingerprint_3.pkl',
#              'tox21_fingerprint_3.pkl',
#              'toxcast_fingerprint_3.pkl',]
DRUG_FILES = ['drug(v0.5).pkl']
PAIR_DIR = '/Data/drugs/ki_ki_pair_final/'  # New pair data for scoring
CKPT_DIR = './results/'
MODEL_NAME = 'test.mdl'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')



# Run settings
argparser = argparse.ArgumentParser()
argparser.register('type', 'bool', str2bool)

argparser.add_argument('--data-path', type=str, default=DATA_PATH,
                       help='Dataset path')
argparser.add_argument('--drug-dir', type=str, default=DRUG_DIR,
                       help='Input drug dictionary')
argparser.add_argument('--drug-files', type=str, default=DRUG_FILES,
                       help='Input drug file')
argparser.add_argument('--pair-dir', type=str, default=PAIR_DIR,
                       help='Input new pairs')
argparser.add_argument('--checkpoint-dir', type=str, default=CKPT_DIR,
                       help='Directory for model checkpoint')
argparser.add_argument('--model-name', type=str, default=MODEL_NAME,
                       help='Model name for saving/loading')
argparser.add_argument('--print-step', type=float, default=100,
                       help='Display steps')
argparser.add_argument('--validation-step', type=float, default=100,
                       help='Number of random search validation')
argparser.add_argument('--train', type='bool', default=True,
                       help='Enable training')
argparser.add_argument('--pretrain', type='bool', default=False,
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
argparser.add_argument('--save-pair-score', type='bool', default=False,
                       help='Save predictions with loaded model')
argparser.add_argument('--top-only', type='bool', default=False,
                       help='Return top/bottom 10% results only')
argparser.add_argument('--embed-d', type = int, default=1,
                       help='0:val task data, 1:v0.n data')

# Train config
argparser.add_argument('--batch-size', type=int, default=32)
argparser.add_argument('--epoch', type=int, default=40)
argparser.add_argument('--learning-rate', type=float, default=1e-3)
argparser.add_argument('--weight-decay', type=float, default=0)
argparser.add_argument('--grad-max-norm', type=int, default=10)
argparser.add_argument('--grad-clip', type=int, default=10)

# Model config
argparser.add_argument('--binary', type='bool', default=False)
argparser.add_argument('--hidden-dim', type=int, default=512)
argparser.add_argument('--drug-embed-dim', type=int, default=300)
argparser.add_argument('--lstm-layer', type=int, default=1)
argparser.add_argument('--lstm-dr', type=float, default=0.0)
argparser.add_argument('--char-dr', type=float, default=0.0)
argparser.add_argument('--bi-lstm', type='bool', default=True)
argparser.add_argument('--linear-dr', type=float, default=0.0)
argparser.add_argument('--char-embed-dim', type=int, default=20)
argparser.add_argument('--s-idx', type=int, default=0)
argparser.add_argument('--rep-idx', type=int, default=2)
argparser.add_argument('--dist-fn', type=str, default='cos')
argparser.add_argument('--seed', type=int, default=None)

#graph
argparser.add_argument('--g_layer', type=int, default = 3)
argparser.add_argument('--g_hidden_dim', type=int, default=512)
argparser.add_argument('--g_out_dim', type=int, default=300)

args = argparser.parse_args()


# Create dirs
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


def run_experiment(model, dataset, run_fn, args):

    # Get dataloaders
    train_loader, valid_loader, test_loader = dataset.get_dataloader(
        batch_size=args.batch_size, s_idx=args.s_idx) 

    # Set metrics
    if args.binary:
        from sklearn.metrics import precision_recall_fscore_support
        metric = partial(precision_recall_fscore_support, average='binary')
        assert args.s_idx == 1
    else:
        metric = np.corrcoef
        assert args.s_idx == 0
    
    # Save embeddings and exit
    if args.save_embed:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, metric, train=False)
        if args.embed_d == 1:
            for drug_file in args.drug_files:
                drugs = pickle.load(open(args.drug_dir + drug_file, 'rb'))
                drugs = drugs.drugs
                save_embed(model, drugs, dataset, args, drug_file)
        else:
            for drug_file in args.drug_files:
                drugs = pickle.load(open(args.drug_dir + drug_file, 'rb'))
                save_embed(model, drugs, dataset, args, drug_file) 
        sys.exit()
    
    # Save predictions on test dataset and exit
    if args.save_prediction:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, metric, train=False)
        save_prediction(model, test_loader, dataset, args)
        sys.exit()

    # Save pair predictions on pretrained model
    if args.save_pair_score:
        model.load_checkpoint(args.checkpoint_dir, args.model_name)
        # run_fn(model, test_loader, dataset, args, metric, train=False)
        save_pair_score(model, args.pair_dir, dataset, args)
        sys.exit()

    # Save and load model during experiments
    if args.train:
        if args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)

        best = 0.0
        converge_cnt = 0
        for ep in range(args.epoch):
            LOGGER.info('Training Epoch %d' % (ep+1))
            run_fn(model, train_loader, dataset, args, metric, train=True)

            if args.valid:
                LOGGER.info('Validation')
                curr = run_fn(model, valid_loader, dataset, args, 
                              metric, train=False)
                if not args.resume and curr > best:
                    best = curr
                    model.save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()},
                        args.checkpoint_dir, args.model_name)
                    converge_cnt = 0
                else:
                    converge_cnt += 1
                
                if converge_cnt >= 3:
                    break
    
    if args.test:
        LOGGER.info('Performance Test on Valid & Test Set')
        if args.train or args.resume:
            model.load_checkpoint(args.checkpoint_dir, args.model_name)
        run_fn(model, valid_loader, dataset, args, metric, train=False)
        run_fn(model, test_loader, dataset, args, metric, train=False)


def get_dataset(path):
    return pickle.load(open(path, 'rb'))


def get_run_fn(args):
    if args.binary:
        return run_bi
    else:
        return run_reg


def get_model(args, dataset):
    dataset.set_rep(args.rep_idx)
    if args.rep_idx == 4:
        model = DrugModel(input_dim=dataset.input_dim,
                          output_dim=1, 
                          hidden_dim=args.hidden_dim,
                          drug_embed_dim=args.drug_embed_dim,
                          lstm_layer=args.lstm_layer,
                          lstm_dropout=args.lstm_dr,
                          bi_lstm=args.bi_lstm,
                          linear_dropout=args.linear_dr,
                          char_vocab_size=len(dataset.char2idx),
                          char_embed_dim=args.char_embed_dim,
                          char_dropout=args.char_dr,
                          dist_fn=args.dist_fn,
                          learning_rate=args.learning_rate,
                          binary=args.binary,
                          is_mlp=False,
                          weight_decay=args.weight_decay,
                          is_graph=True,
                          g_layer=args.g_layer,
                          g_hidden_dim=args.g_hidden_dim,
                          g_out_dim=args.g_out_dim).cuda()

    else:
        model = DrugModel(input_dim=dataset.input_dim,
                          output_dim=1, 
                          hidden_dim=args.hidden_dim,
                          drug_embed_dim=args.drug_embed_dim,
                          lstm_layer=args.lstm_layer,
                          lstm_dropout=args.lstm_dr,
                          bi_lstm=args.bi_lstm,
                          linear_dropout=args.linear_dr,
                          char_vocab_size=len(dataset.char2idx),
                          char_embed_dim=args.char_embed_dim,
                          char_dropout=args.char_dr,
                          dist_fn=args.dist_fn,
                          learning_rate=args.learning_rate,
                          binary=args.binary,
                          is_mlp=args.rep_idx > 1,
                          weight_decay=args.weight_decay,
                          is_graph=False,
                          g_layer=None,
                          g_hidden_dim=None,
                          g_out_dim=None).cuda()
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
        args.checkpoint_dir + 'logs/' + args.model_name + '.txt', 'w')
    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)


def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def init_parameters(args, model_name, model_idx):
    args.model_name = '{}-{}'.format(model_name, model_idx)
    args.learning_rate = np.random.uniform(1e-4, 2e-3)


def main():
    # Initialize logging and prepare seed
    init_logging(args)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))

    # Get datset, run function, model
    dataset = get_dataset(args.data_path)
    run_fn = get_run_fn(args)
    model_name = args.model_name

    # Random search validation
    for model_idx in range(args.validation_step):
        LOGGER.info('Validation step {}'.format(model_idx+1))
        init_seed(args.seed)
        init_parameters(args, model_name, model_idx)
        LOGGER.info(args)

        # Get model
        model = get_model(args, dataset)

        # Run experiment
        run_experiment(model, dataset, run_fn, args)


if __name__ == '__main__':
    main()
