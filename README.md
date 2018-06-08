# ReSimNet
A Pytorch Implementation of paper
> ReSimNet: Drug Response Similarity Prediction based on Siamese Neural Network <br>
> Jeon and Park et al., 2018

## Abstract
As more data available, it has become possible to data-driven new drug discovery pipeline. We, therefore, propose a model for predicting a similarity score between two drugs based on the differential gene expression patterns of the two drugs. Our model trains a Siamese neural network that takes fingerprints of a pair of two drugs and predicts its similarity score.

## Pipeline
![Full Pipeline](/images/pipeline_updated_kang2.png)

## Requirements
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downlaods)
- Install [cudnn-v5.1](https://developer.nvidia.com/cudnn)
- Install [Pytorch 0.3.0](https://pytorch.org/)
- Python version >= 3.4.3 is required

## CMap Score Prediction using ReSimNet
For your own fingerprint pairs, ReSimNet provides a predicted CMap Score for each pair. Running download.sh and test.sh will first download pretrained ReSimNet with sample datasets, and save a result file for predicted CMap scores.
```bash
# Download datasets and pretrained model from S3
$. download.sh

# Save scores of sample pair data
$. predict.sh
```
- Input Fingerprint pair file must be a .csv file in which every row consists of two columns denoting two Fingerprints of each pair (please, place files under './tasks/data/pairs/').
- Predicted CMap scores will be at each row of a file './results/input-pair-file.model-name.csv'.

```bash
# Sample results
prediction
0.9146181344985962
0.9301251173019409
0.8519644737243652
0.9631381034851074
0.7272981405258179
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
