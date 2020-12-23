#!/bin/bash

parallel python random_swaps.py --src-file ../data/aa_fc_grouped/data.txt --tgt-file ../data/aa_fc_sampled_limited_uneven_rand.{1}/data.txt --chunk-size 500 --max-source-size 100000 --no-keep-all --not-always-activated --epochs T5 T6 T7 ::: $(seq -w 1 100)

parallel cp ../data/aa_fc_grouped/features.txt ../data/aa_fc_sampled_limited_uneven_rand.{1}/features.txt ::: $(seq -w 1 100)
parallel cp ../data/aa_fc_grouped/vocab.txt ../data/aa_fc_sampled_limited_uneven_rand.{1}/vocab.txt ::: $(seq -w 1 100)
