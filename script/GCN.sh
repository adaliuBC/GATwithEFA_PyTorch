#!/bin/bash

python ./train.py \
    --dataset=fcinkml \
    --model=GCN \
    --hidden=32 \
    --dropout=0.1 \
	--alpha=0.1 \
	--weight_decay=0 \
    --epochs=200	\
    --lr=0.01 \
    --batch_size=8 \
    --val_freq=1 \
    --patience=200
cp ./log.txt /data/pyGAT-master
cp ./log.txt /data/