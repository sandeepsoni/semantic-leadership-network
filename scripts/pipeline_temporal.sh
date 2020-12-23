#!/bin/bash

### Assume that the embeddings have been trained at the beginning of this script

## Find the nearest neighbors
# Note that this step requires a few hours
#parallel python near_neighbors.py --dir-path ../data/aa_temp_grouped --embeddings-file out.embeddings --near-neighbors-file {1}.neighbors.pkl --facet-name {1} --nearest 25 ::: MAIN T0 T1 T2 T3 T4 T5 T6 T7 T8 T9

## Find the semantic changes according to the "local" metric for change
#python local_dynamic_ranks.py --dir-path ../data/aa_temp_grouped --embeddings-file out.embeddings --scores-file local.scores

## Find all the frequency files
#python get_frequency_stats.py --dir-path ../data/aa_temp_grouped --data-file data.txt --word-freq-file words.freq --doc-freq-file docs.freq --period-freq-file periods.freq --source-freq-file sources.freq --periods 10

## Name identification filter (to be run separately and just once)
#python name_identification.py --dir-path ../data/aa_temp_grouped --scores-file local.scores --embeddings-file out.embeddings --names-file names.ignore --casestats-file ../annotations/wordcase.stats --annotations-file ../annotations/names_locations.csv

## Create all other filters
python create_filters.py --dir-path ../data/aa_temp_grouped --periods-freq-file periods.freq --sources-freq-file sources.freq --scores-file local.scores

## Apply all the filters
python filter_words.py --dir-path ../data/aa_temp_grouped --scores-file local.scores --period-freq-file periods.freq --filter-names functional-words end-hyphens very-short infrequent-periods infrequent-sources trivial-change-freq trivial-neighbor-freq-words names many-names-neighbors --output-file words.csv
