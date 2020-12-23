#!/bin/bash

### Now apply all the filters to randomized data 
source constants.sh

#parallel --jobs 16 python leadership_scores.py --src-path ../data/$RAND_PREFIX.{1} --temp-path ../data/aa_temp_grouped --embeddings-file out.embeddings --feats-file features.txt --changes-file words.csv --leaders-file leaders.csv --lead-types l1 ::: $(seq -f "%03g" 1 100) 
#parallel --jobs 16 python leadership_thresholds.py --src-path ../data/$RAND_PREFIX.{1} --leaders-file leaders.csv --thresh {2} --thresholds-file leaders.{2}.{3}.csv --lead-type {3} ::: $(seq -f "%03g" 1 100) ::: 0 ::: l1
#parallel --jobs 16 python leadership_stats.py --src-path ../data/$RAND_PREFIX.{1} --leaders-file leaders.{2}.{3}.csv --leader-stats-file leader_stats.{2}.{3}.csv --leader-follower-stats-file leader_follower_stats.{2}.{3}.csv ::: $(seq -f "%03g" 1 100) ::: 0 ::: l1

#parallel python aggregate_leader_runs.py --dir-prefix ../data/$RAND_PREFIX --output-file ../data/aa_fc_sampled_limited_rand_uneven_aggregate/leader_probs.{1}.{2}.csv --nums $(seq -f "%03g" 1 100) --thresholds {1} --lead-types {2} --column-num 2 ::: 0 ::: l1

### A slightly different way of comparing against randomized datasets

parallel --jobs 16 python leadership_scores_new.py --rand-path ../data/$RAND_PREFIX.{1} --obs-path ../data/aa_fc_grouped --embeddings-file out.embeddings --feats-file features.txt --leaders-file leaders.{2}.{3}.csv --output-file leaders_random.{2}.{3}.csv ::: $(seq -f "%03g" 1 100) ::: 0 ::: l1
