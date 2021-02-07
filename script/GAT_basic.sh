#!/bin/bash

python ./train.py \
    --dataset=fcinkml \
    --model=GAT \
    --residual \
    --nlayer=7 \
    --hidden=32 \
    --alpha=0.2 \
    --dropout=0.1 \
    --nb_heads=8 \
    --nout_heads=2 \
    --epochs=200 \
    --lr=0.005 \
    --batch_size=3 \
    --val_freq=1 \
    --patience=30
cp ./log.txt /data/