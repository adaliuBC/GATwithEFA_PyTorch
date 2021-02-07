#!/bin/bash

python ./train.py \
    --dataset=fcinkml \
    --model=MLP \
    --hidden=32 \
    --dropout=0.2 \
    --weight_decay=0 \
    --epochs=200 \
    --lr=0.001 \
    --batch_size=8 \
    --val_freq=1 \
    --patience=50 
cp ./log.txt /data/pyGAT-master
