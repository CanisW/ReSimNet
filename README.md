# DrugResponse2Vec
A Pytorch Implementation of paper
> DrugResponse2vec: Representation of Drug in Vector Space based on Drug-Cell line Response <br>
> Jeon and Park et al., 2018

## Abstract
As more data available, it has become possible to data-driven new drug discovery pipeline. We, therefore, propose a model for predicting a similarity score between two drugs based on the differential gene expression patterns of the two drugs. Our model trains a Siamese neural network that takes fingerprints of a pair of two drugs and predicts its similarity score.

## Project Structure
**./main.py** : Run experiments with arguments. <br>
**./tasks/drug_run.py** : Functions for running experiments. <br>
**./tasks/drug_task.py** : Preprocess datasets. <br>
**./models/drug_model.py** : LSTM Siamese network. <br>

## Run Code
### Preprocess dataset
```
$ python3 main.py
```
### Train/test LSTM Siamese network.
```
$ python3 main.py
```

## Experimental Results
### Regression
#### DrugResponse2vec
Testset Split|Overall CorrCoef|Top/Bottom 1% CorrCoef|AUROC
-------------|----------------|------------- --------|-----
Total | 0.447 | 0.877 | 0.693
KK | 0.606 | 0.932 | 0.777
KU | 0.340 | 0.737 | 0.639
UU | 0.120 | 0.240 | 0.555


### Binary Classification

## Liscense
