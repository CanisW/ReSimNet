# DrugResponse2Vec
A Pytorch Implementation of paper
> DrugResponse2vec: Representation of Drug in Vector Space based on Drug-Cell line Response <br>
> Jeon and Park et al., 2018

## Abstract
As more data available, it has become possible to data-driven new drug discovery pipeline. We, therefore, propose a model for predicting a similarity score between two drugs based on the differential gene expression patterns of the two drugs. Our model trains a Siamese neural network that takes fingerprints of a pair of two drugs and predicts its similarity score.

## Project Structure
./main.py : Run experiments with arguments.
./tasks/drug_run.py : Functions for running experiments.
./tasks/drug_task.py : Preprocess datasets.
./models/drug_model.py : LSTM Siamese network model.

## Run Code
```
$ python3 main.py
```


## Experimental Results
### Regression
aaa | aaa
--- | ---
fff | fff
fff | sddd

### Binary Classification

## Liscense
