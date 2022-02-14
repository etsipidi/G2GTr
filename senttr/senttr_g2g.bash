#!/bin/bash
MAIN_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/embeddings/"
BERT_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/tempbert/"
DATA_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/data/"

python run.py train --lr 1e-5 --lr2 1e-4 -w 0.001 --modelname senttr_g2g --batch_size 32 --buckets 5 \
         --ftrain $DATA_PATH/train.conll \
         --ftrain_seq $DATA_PATH/oracle_eager.seq \
         --ftest $DATA_PATH/test.conll \
         --fdev $DATA_PATH/dev.conll \
         --bert_path $BERT_PATH --punct --n_attention_layer 6 --epochs 12 \
         --input_graph --act_thr 180 --use_two_opts --main_path $MAIN_PATH --device 0
