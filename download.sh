#!/bin/bash

# Create data folder and download preprocessed file
cd tasks
mkdir -p data
cd data
wget https://s3-us-west-2.amazonaws.com/resimnet/drug\(v0.6\).pkl

# Download sample pair data
mkdir -p pairs
cd pairs
wget https://s3-us-west-2.amazonaws.com/resimnet/examples.csv
cd ../../..

# Create checkpoint folder
mkdir -p results
cd results
mkdir -p logs

# Download pretrained ReSimNet
wget https://s3-us-west-2.amazonaws.com/resimnet/resimnet_pretrained
cd ..

