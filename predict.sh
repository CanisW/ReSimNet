#!/bin/bash
# predict pair scores when given with two input drud_ids.
# calculate prediction scores based on averged scores of all 10 models.
# if you do not want this, set --save-pair-score-ensemble to false
python main.py --save-pair-score true --save-pair-score-ensemble true --pair-dir './tasks/data/pairs/' --fp-dir './tasks/data/fingerprint_v0.6_py2.pkl' --data-path './tasks/data/drug(v0.6).pkl' --model-name 'ReSimNet.mdl' --rep-idx 2
