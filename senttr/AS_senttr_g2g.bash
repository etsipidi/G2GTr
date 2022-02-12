#!/bin/bash
MAIN_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/embeddings"
BERT_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/tempbert/"
DATA_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/data/projective"

python run.py train --lr 1e-5 --lr2 1e-4 -w 0.001 --modelname arcstandard_senttr_g2g --batch_size 40 --buckets 10 \
         --ftrain $DATA_PATH/train.conll \
         --ftrain_seq $DATA_PATH/oracle_standard.seq \
         --ftest $DATA_PATH/test.conll \
         --fdev $DATA_PATH/dev.conll \
         --bert_path $BERT_PATH --punct --n_attention_layer 7 --epochs 12 \
         --input_graph --act_thr 210 --use_two_opts --main_path $MAIN_PATH --device 0
