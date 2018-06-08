#!/bin/bash

# Create data folder
cd tasks
mkdir -p data
cd data
mkdir -p drug

# Download sample data and preprocess 

cd ../..

# Create checkpoint folder
mkdir -p results
cd results
mkdir -p logs

# Download pretrained 
wget https://s3-us-west-2.amazonaws.com/resimnet/fingerprint_cos
cd ..

