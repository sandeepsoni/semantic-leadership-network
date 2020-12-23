#!/bin/bash

### Apply all the filters to real data
python leadership_scores.py --src-path ../data/aa_fc_grouped --temp-path ../data/aa_temp_grouped --embeddings-file out.embeddings --feats-file features.txt --changes-file words.csv --leaders-file leaders.csv --lead-types l1
parallel python leadership_thresholds.py --src-path ../data/aa_fc_grouped --leaders-file leaders.csv --thresh {1} --thresholds-file leaders.{1}.{2}.csv --lead-type {2} ::: 0 ::: l1
parallel python leadership_stats.py --src-path ../data/aa_fc_grouped --leaders-file leaders.{1}.{2}.csv --leader-stats-file leader_stats.{1}.{2}.csv --leader-follower-stats-file leader_follower_stats.{1}.{2}.csv ::: 0 ::: l1
