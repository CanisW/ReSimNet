# ReSimNet
A Pytorch Implementation of paper
> ReSimNet: Drug Response Similarity Prediction based on Siamese Neural Network <br>
> Jeon and Park et al., 2018

## Abstract
As more data available, it has become possible to data-driven new drug discovery pipeline. We, therefore, propose a model for predicting a similarity score between two drugs based on the differential gene expression patterns of the two drugs. Our model trains a Siamese neural network that takes fingerprints of a pair of two drugs and predicts its similarity score.

## Pipeline
![Full Pipeline](/images/pipeline_updated_kang2.png)

## Requirement
* Python 3.4.3
* Pytorch 0.3.0

## Download dataset & pretrained model
```bash
# Download dataset and pretrained model from S3
$ . download.sh
```

## Test model
```bash
# Test with default setting
$ . test.sh
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
