"""
python final_words.py --dir-path ../data/aa_temp.1/ --changes-file words.changes --output-file final.changes
"""

import argparse
import os
import pandas as pd
from copy import deepcopy
from helpful_functions import init_dict, period_names

def readArgs ():
	parser = argparse.ArgumentParser (description="filter the words")
	parser.add_argument ("--dir-path", required=True, type=str, help="directory path")
	parser.add_argument ("--changes-file", required=True, type=str, help="changes file")
	parser.add_argument ("--output-file", required=True, type=str, help="output file")
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
	df = pd.read_csv (os.path.join (args.dir_path, args.changes_file), sep=";")
	rows = df[["word", "rank", "Period1", "Period2", "Neighbors1", "Neighbors2", "Freq1", "Freq2", "NeighborsFreq1", "NeighborsFreq2"]].values.tolist()
	new_rows = list ()
	for row in rows:
		p1, p2 = row[2], row[3]
		t1, t2 = int(p1[1:]), int(p2[1:])
		if (t2 - t1) == 1:
			new_rows.append (row)
	
	new_df = pd.DataFrame (new_rows, columns=["word", "rank", "Period1", "Period2", "Neighbors1", "Neighbors2", "Freq1", "Freq2", "NeighborsFreq1", "NeighborsFreq2"])
	new_df.to_csv (os.path.join (args.dir_path, args.output_file), sep=";", header=True, index=False)

if __name__ == "__main__":
	main (readArgs ())
