# ReSimNet
A Pytorch Implementation of paper
> ReSimNet: Drug Response Similarity Prediction based on Siamese Neural Network <br>
> Jeon and Park et al., 2018

## Abstract
As more data available, it has become possible to data-driven new drug discovery pipeline. We, therefore, propose a model for predicting a similarity score between two drugs based on the differential gene expression patterns of the two drugs. Our model trains a Siamese neural network that takes fingerprints of a pair of two drugs and predicts its similarity score.

## Pipeline
![Full Pipeline](/images/pipeline_updated_kang2.eps)

## Project Structure
**./main.py** : Run experiments with arguments. <br>
**./tasks/drug_run.py** : Functions for running experiments. <br>
**./tasks/drug_task.py** : Preprocess datasets. <br>
**./models/drug_model.py** : LSTM Siamese network. <br>

## Run Code
### Requirement
* Python 3.4.3
* Pytorch 0.3.0

### Preprocess dataset
Before preprocessing, you should create a directory ./tasks/data/drug/ and unzip files downloaded from [here](https://google.com).
```
# This will create a pickle file ./data/drug/drug(tmp).
$ cd tasks/
$ python3 drug_task.py
```
### Train/test LSTM Siamese network.
```
usage: main.py [-h] [--data-path DATA_PATH] [--drug-dir DRUG_DIR]
               [--drug-files DRUG_FILES] [--pair-dir PAIR_DIR]
               [--checkpoint-dir CHECKPOINT_DIR] [--model-name MODEL_NAME]
               [--print-step PRINT_STEP] [--validation-step VALIDATION_STEP]
               [--train TRAIN] [--pretrain PRETRAIN] [--valid VALID]
               [--test TEST] [--resume RESUME] [--debug DEBUG]
               [--save-embed SAVE_EMBED] [--save-prediction SAVE_PREDICTION]
               [--save-pair-score SAVE_PAIR_SCORE] [--top-only TOP_ONLY]
               [--embed-d EMBED_D] [--batch-size BATCH_SIZE] [--epoch EPOCH]
               [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY]
               [--grad-max-norm GRAD_MAX_NORM] [--grad-clip GRAD_CLIP]
               [--binary BINARY] [--hidden-dim HIDDEN_DIM]
               [--drug-embed-dim DRUG_EMBED_DIM] [--lstm-layer LSTM_LAYER]
               [--lstm-dr LSTM_DR] [--char-dr CHAR_DR] [--bi-lstm BI_LSTM]
               [--linear-dr LINEAR_DR] [--char-embed-dim CHAR_EMBED_DIM]
               [--s-idx S_IDX] [--rep-idx REP_IDX] [--dist-fn DIST_FN]
               [--seed SEED] [--g_layer G_LAYER] [--g_hidden_dim G_HIDDEN_DIM]
               [--g_out_dim G_OUT_DIM] [--g_dropout G_DROPOUT]

# Train/test Siamese network on Fingerprint dataset (regression)
$ python3 main.py --rep-idx 2 --train True --test True
```

## Experimental Results
Testset Split|Correlation|MSE (Total / 1% / 2% / 5%)|AUROC
-------------|----------------|----------------------|-----
Total | 0.447 | 0.091 / 0.012 / 0.015 / 0.028 | 0.693
KK | 0.606 | 0.072 / 0.008 / 0.007 / 0.008 | 0.777
KU | 0.340 | 0.102 / 0.029 / 0.039 / 0.044 | 0.639
UU | 0.120 | 0.114 / 0.048 / 0.074 / 0.117 | 0.555


## Liscense
Apache License 2.0
