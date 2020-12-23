import argparse
import pandas as pd
import os
import itertools
import math
import ujson
import numpy as np
import logging
logging.basicConfig (format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
from helpful_functions import readEmbeddings, normalize, sigmoid, logsigmoid, MAIN_FEAT

"""
Example:
python context_gathering.py --src-path ../data/aa_fc_grouped --data-file data.txt --temp-path ../data/aa_temp_grouped/ --changes-file words.csv --contexts-file word_contexts.jsonl
"""

def readArgs ():
	parser = argparse.ArgumentParser (description="leadership score calculation")
	parser.add_argument ("--src-path", type=str, required=True, help="directory that contains the source embeddings")
	parser.add_argument ("--data-file", type=str, required=True, help="file contains all the data in one file")
	parser.add_argument ("--temp-path", type=str, required=True, help="directory that contains the temporal embeddings and other files")
	parser.add_argument ("--changes-file", type=str, required=True, help="file contains all the changes")
	parser.add_argument ("--contexts-file", type=str, required=True, help="file that contains the target and context words")
	args = parser.parse_args ()
	return args

def words2dict (all_words):
	""" Every item in the list `all_words` is of the form [word, period1, period2]

		Turn into dictionary where word is a key and (period1, period2) is a pair of value
	"""
	words = dict ()
	for item in all_words:
		word, p1, p2 = item[0], item[1], item[2]
		words[word] = (p1, p2)

	return words	

def records_iterator (filename):
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split ("\t")
			period, source, text = parts[1], parts[2].split("_")[1], parts[3]
			yield text, period, source

def if_correct_period (period, end_points):
	p1, p2 = end_points
	t = period[1]
	t1, t2 = p1[1], p2[1]
	return t > t1 and t <= t2

def get_contexts (text, period, source, changes, window_size=5):
	items = list ()
	tokens = text.split ()
	for i, token in enumerate (tokens):
		if token in changes and if_correct_period (period, changes[token]):
			lc = tokens[max(0, i-window_size):i]
			rc = tokens[min(len (tokens), i+1): min(len (tokens), i+1+window_size)]
			items.append ({"word": token, \
						   "period": period, \
						   "source": source, \
						   "left_contexts": lc, \
						   "right_contexts": rc})

	return items	

def main (args):	
	# Let's first read the word changes to memory
	changes_file = os.path.join (args.temp_path, args.changes_file)
	candidates = pd.read_csv (changes_file, sep=";")
	all_words = candidates[["word", "Period1", "Period2"]].values.tolist()
	changes = words2dict (all_words)

	# Now let's read a document one at a time
	with open (os.path.join (args.src_path, args.contexts_file), "w") as fout:
		for text, period, source in records_iterator (os.path.join (args.src_path, args.data_file)):
			items = get_contexts (text, period, source, changes)
			for item in items:
				fout.write (ujson.dumps (item) + "\n")
			
if __name__ == "__main__":
	main (readArgs ())
