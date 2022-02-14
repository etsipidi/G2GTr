#!/bin/bash
MAIN_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/embeddings/"
BERT_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/tempbert/"
DATA_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/data/projective/"

python run.py train --lr 1e-5 -w 0.001 --modelname senttr --batch_size 1 --buckets 1 \
         --ftrain $DATA_PATH/train.conll \
         --ftrain_seq $DATA_PATH/oracle_eager.seq \
         --ftest $DATA_PATH/test.conll \
         --fdev $DATA_PATH/dev.conll \
         --bert_path $BERT_PATH --punct --n_attention_layer 6 --epochs 12 --act_thr 210 \
         --main_path $MAIN_PATH --device 0 -t 1
