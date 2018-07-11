# ReSimNet
A Pytorch Implementation of paper
> ReSimNet: Drug Response Similarity Prediction based on Siamese Neural Network <br>
> Jeon and Park et al., 2018

## Abstract
Two important things in the new drug discovery pipeline are identifying a suitable target for a disease and finding a molecule that binds to the target. Once a target for the disease is identified, chemical compounds that can bind to the target are found through high throughput screening. Structural analogs of the drugs that bind to the target have also been selected as drug candidates. However, even though compounds are not structural analogs, they may achieve the desired response and these candidate compounds may be used for the disease. A new drug discovery method based on drug response, and not on drug structure, is necessary; therefore, we propose a drug response-based drug discovery model called ReSimNet.

We implemented a Siamese neural network that receives the structures of two chemical compounds as an input and trains the similarity of the differential gene expression patterns of the two chemical compounds. ReSimNet can predict the transcriptional response similarity between a pair of chemical compounds and find compound pairs that are similar in response even though they may have dissimilar structures. ReSimNet outperforms structure-based representations in predicting the drug response similarity of compound pairs. Precisely, ReSimNet obtains 0.447 of Pearson correlation (p-value < 10^-6) and 0.967 of Precision@1% when we compare predicted similarity scores and actual transcriptional response-based similarity scores obtained from Connectivity Map. In addition, for the qualitative analysis, we test ReSimNet on the ZINC15 dataset and show that ReSimNet successfully identifies chemical compounds that are relevant to the well-known drugs.

## Pipeline
![Full Pipeline](/images/pipeline_updated_kang2.png)

## Requirements
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downlaods)
- Install [cudnn-v5.1](https://developer.nvidia.com/cudnn)
- Install [Pytorch 0.3.0](https://pytorch.org/)
- Python version >= 3.4.3 is required

## CMap Score Prediction using ReSimNet
For your own fingerprint pairs, ReSimNet provides a predicted CMap score for each pair. Running download.sh and predict.sh will first download pretrained ReSimNet with sample datasets, and save a result file for predicted CMap scores.
```bash
# Clone repository
$ git clone https://github.com/jhyuklee/ReSimNet.git
$ cd ReSimNet/

# Download datasets and pretrained model from S3
$ bash download.sh

# Save scores of sample pair data
$ bash predict.sh
```
Input Fingerprint pair file must be a .csv file in which every row consists of two columns denoting two Fingerprints of each pair. Please, place files under './tasks/data/pairs/'. 
```bash
# Sample Fingerprints (./tasks/data/pairs/examples.csv)
drug_A,drug_B
000000...01000,00010...01000
000000...00010,00000...00100
010000...10000,00000...00000
000100...01000,00001...01000
000000...00000,01000...10000
```
Predicted CMap scores will be saved at each row of a file './results/input-pair-file.model-name.csv'.
```bash
# Sample results (./results/examples.csv.resimnet_pretrained.csv')
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
