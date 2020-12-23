"""
python filter_words.py --dir-path ../data/aa_temp.1 --scores-file local.scores --period-freq-file periods.freq --filter-names end-hyphens very-short infrequent-periods infrequent-sources trivial-change-freq trivial-neighbor-freq-words names --output-file words.csv
"""

import argparse
import os
import pandas as pd
from copy import deepcopy
from helpful_functions import init_dict

def readArgs ():
	parser = argparse.ArgumentParser (description="filter the words")
	parser.add_argument ("--dir-path", required=True, type=str, help="directory path")
	parser.add_argument ("--scores-file", required=True, type=str, help="scores file")
	parser.add_argument ("--period-freq-file", required=True, type=str, help="period frequency file")
	parser.add_argument ("--filter-names", required=True, type=str, nargs="+", help="filter names")
	parser.add_argument ("--output-file", required=True, type=str, help="output file")
	parser.add_argument ("--periods", required=False, type=int, default=10, help="number of periods")
	args = parser.parse_args ()
	return args

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

def main (args):
	df = pd.read_csv (os.path.join (args.dir_path, args.scores_file), sep=";")
	words = set ()
	for filter_name in args.filter_names:
		with open (os.path.join (args.dir_path, f"{filter_name}.ignore")) as fin:
			for line in fin:
				words.add (line.strip())

	period_names = [f'T{i}' for i in range (args.periods)]

	period_freq = readDictFromFile (os.path.join (args.dir_path, args.period_freq_file), period_names)

	df = df[~df["word"].isin (words)]
	rows = df[["word", "rank", "Period1", "Period2", "Neighbors1", "Neighbors2"]].values.tolist()
	new_rows = deepcopy(rows)
	for row in new_rows:
		word = str (row[0])
		p1, p2 = row[2], row[3]
		n1, n2 = eval(row[4]), eval(row[5])
		#print (word, p1, p2)
		row.append (period_freq[word][p1])
		row.append (period_freq[word][p2])
		row.append ([period_freq[n][p1] for n in n1])
		row.append ([period_freq[n][p2] for n in n2])
	
	new_df = pd.DataFrame (new_rows, columns=["word", "rank", "Period1", "Period2", "Neighbors1", "Neighbors2", "Freq1", "Freq2", "NeighborsFreq1", "NeighborsFreq2"])
	new_df.to_csv (os.path.join (args.dir_path, args.output_file), sep=";", header=True, index=False)

if __name__ == "__main__":
	main (readArgs ())
