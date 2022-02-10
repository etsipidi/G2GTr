#!/bin/bash
MAIN_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/embeddings"
BERT_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/tempbert/"
DATA_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/data"

python run.py train --lr1 1e-5 --lr2 1e-4 -w 0.001 --modelname arceager_senttr_g2g --batch_size 40 --buckets 10 \
         --ftrain $DATA_PATH/train.conllx \
         --ftrain_seq $DATA_PATH/oracle-eager.seq \
         --ftest $DATA_PATH/test.conllx \
         --fdev $DATA_PATH/dev.conllx \
         --bert_path $BERT_PATH --punct --n_attention_layer 7 --epochs 12 \
         --input_graph --act_thr 210 --use_two_opts --main_path $MAIN_PATH
