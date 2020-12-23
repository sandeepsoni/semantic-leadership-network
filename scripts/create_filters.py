"""
python create_filters.py --dir-path ../data/aa_temp.1/ --periods-freq-file periods.freq --sources-freq-file sources.freq --scores-file local.scores

Assume that ../data/aa_temp.1/names.ignore file is already present.
"""

import argparse
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from helpful_functions import is_count_greater, is_count_smaller, source_names, init_dict

def readArgs ():
	parser = argparse.ArgumentParser (description="filter words that change the most")
	parser.add_argument ("--dir-path", required=True, type=str, help="directory that contains the required files")
	parser.add_argument ("--periods-freq-file", required=True, type=str, help="file that contains period frequencies")
	parser.add_argument ("--sources-freq-file", required=True, type=str, help="file that contains source frequencies")
	parser.add_argument ("--scores-file", required=True, type=str, help="file contains near neighbors and scores")
	parser.add_argument ("--periods", required=False, type=int, default=10, help="number of periods")
	return parser.parse_args ()

def filter_by_freq (period_frequencies, topn=250):
	frequencies = {w: sum(period_frequencies[w].values()) for w in period_frequencies}
	ranked_words = [w for w,_ in sorted (frequencies.items(), key=lambda x:x[1], reverse=True)]
	return set (ranked_words[0:topn])

def filter_by_period_freq (period_frequencies, period_thresh=5):
	""" Return True if more than `period_thresh` periods have zero counts"""
	zero_freq_dist = [period_frequencies[period] == 0 for period in period_frequencies]
	return is_count_greater (zero_freq_dist, thresh=period_thresh)


def filter_by_source_freq (source_frequencies, source_thresh=4):
	""" Return True if less than `source_thresh` sources have non-zero counts"""
	nonzero_freq_dist = [source_frequencies[source] > 0 for source in source_frequencies]
	return is_count_smaller (nonzero_freq_dist, source_thresh)

def readDictFromFile (filename, names):
	freq = dict ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split (",")
			word, facet, count = parts[0], parts[1], parts[2]
			if word not in freq:
				freq[word] = init_dict (names)
			freq[word][facet] = int (count)
	return freq

def write_filters (words, dir_path, filter_name):
	with open (os.path.join (dir_path, f"{filter_name}.ignore"), "w") as fout:
		for word in words:
			fout.write (f"{word}\n")

def main (args):
	ranks = pd.read_csv (os.path.join (args.dir_path, args.scores_file), sep=";")
	words = {str(w) for w in ranks["word"].values}

	period_names = [f'T{i}' for i in range (args.periods)]
	# remove words that have hyphens at the end
	hyphen_endings = {w for w in words if w.endswith ("-")}
	# remove words that are too short
	very_short_words = {w for w in words if len(w) <=2}

	# remove words with zero counts in more than 5 periods
	period_freq_vocab = readDictFromFile (os.path.join (args.dir_path, args.periods_freq_file), period_names)

	# remove words that are very high frequency (i.e functional words)
	functional_words = filter_by_freq (period_freq_vocab, topn=250)
	infrequent_period_words = {w \
							   for w in period_freq_vocab \
							   if filter_by_period_freq (period_freq_vocab[w], period_thresh=5) and w in words}

	# remove words if they have non-zero counts only in 3 or fewer sources.
	source_freq_vocab = readDictFromFile (os.path.join (args.dir_path, args.sources_freq_file), source_names)
	infrequent_source_words = {w \
							   for w in source_freq_vocab \
							   if filter_by_source_freq (source_freq_vocab[w], source_thresh=4) and w in words}


	#remove words with trivial counts in the periods they are supposed to have changed.
	trivial_freq_words = set()
	rows = ranks[["word", "Period1", "Period2"]].values
	for row in rows:
		w, p1, p2 = str(row[0]), row[1], row[2]
		if period_freq_vocab[w][p1] <= 10 or period_freq_vocab[w][p2] <= 10:
			trivial_freq_words.add (w)

	# remove words if many neighbors with trivial counts in the periods words supposed to have changed.
	trivial_neighbor_freq_words = set ()
	rows = ranks[["word", "Period1", "Period2", "Neighbors1", "Neighbors2"]].values
	for row in rows:
		w, p1, p2, n1, n2 = str (row[0]), row[1], row[2], eval(row[3]), eval(row[4])
		n1_freq = [period_freq_vocab[n][p1] for n in n1]
		n2_freq = [period_freq_vocab[n][p2] for n in n2]

		if is_count_greater ([f <= 3 for f in n1_freq], 7) or is_count_greater ([f <= 3 for f in n2_freq], 7):
			trivial_neighbor_freq_words.add (w)

	# remove words if many neighbors are names in the periods when words are supposed to have changed.
	with open (os.path.join (args.dir_path, "names.ignore")) as fin:
		names = {line.strip() for line in fin}

	too_many_neighbors_as_names = set ()
	rows = ranks[["word", "Period1", "Period2", "Neighbors1", "Neighbors2"]].values
	for row in rows:
		w, p1, p2, n1, n2 = str (row[0]), row[1], row[2], eval(row[3]), eval(row[4])
		n1_names = [n for n in n1 if n in names]
		n2_names = [n for n in n2 if n in names]

		if len (n1_names) >= 5 or len (n2_names) >= 5:
			too_many_neighbors_as_names.add(w)
	

	write_filters (functional_words, args.dir_path, "functional-words")
	write_filters (hyphen_endings, args.dir_path, "end-hyphens")        
	write_filters (very_short_words, args.dir_path, "very-short")        
	write_filters (infrequent_period_words, args.dir_path, "infrequent-periods")
	write_filters (infrequent_source_words, args.dir_path, "infrequent-sources")
	write_filters (trivial_freq_words, args.dir_path, "trivial-change-freq")
	write_filters (trivial_neighbor_freq_words, args.dir_path, "trivial-neighbor-freq-words")
	write_filters (too_many_neighbors_as_names, args.dir_path, "many-names-neighbors")

if __name__ == "__main__":
	main (readArgs ())
