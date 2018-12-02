#!/bin/bash
python main.py --data-path './tasks/data/drug(v0.6).pkl' --model-name 'test.mdl' --rep-idx 2 --perform-ensemble True 

# Run
# 0 : smiles, 1: inchikey, 2: ecfp, 3: mol2vec
