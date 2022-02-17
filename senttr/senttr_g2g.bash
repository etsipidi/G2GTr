#!/bin/bash
MAIN_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/"
BERT_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/tempbert/"
DATA_PATH="/home/etsipidi/Documents/Parsing/G2GTr/senttr/data/ud_v1_3/"

python run.py train --lr 1e-5 --lr2 1e-4 -w 0.001 --modelname senttr_st_g2g --batch_size 40 --buckets 10 \
         --ftrain $DATA_PATH/train.conll \
         --ftrain_seq $DATA_PATH/oracle_standard.seq \
         --ftest $DATA_PATH/test.conll \
         --fdev $DATA_PATH/dev.conll \
         --bert_path $BERT_PATH --punct --n_attention_layer 7 --epochs 12 \
         --model model/ --vocab model/ \
         --input_graph --act_thr 210 --use_two_opts --main_path $MAIN_PATH --device 0 -t 8
