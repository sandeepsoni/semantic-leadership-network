""" Python script to generate relevant files that can be used by the model.

Ex.
time python makeAAOneFile.py --data-dir /hg191/corpora/clean-accessible-v4.0/ --data-file /hg191/sandeep/projects/geoSGLM/aa_fully_conditional/data.txt --features-file /hg191/sandeep/projects/geoSGLM/aa_fully_conditional/features.txt --vocab-file /hg191/sandeep/projects/geoSGLM/aa_fully_conditional/vocab.txt --mode source --epochs 10
/hg191/sandeep/projects/geoSGLM/aa_fully_conditional/data.txt contains 191206 records

real    5m31.524s
user    5m24.288s
sys     0m4.873s

"""
import argparse
import os
import datetime
import re
import ujson
from collections import defaultdict
from helpful_functions import safe_open_w

def readArgs ():
	parser = argparse.ArgumentParser (description="")
	parser.add_argument ("--data-dir", required=True, help="data directory that contains the files")
	parser.add_argument ("--data-file", required=True, help="file in which the data is collated")
	parser.add_argument ("--features-file", required=True, help="file in which the categorical features are listed")
	parser.add_argument ("--vocab-file", required=True, help="file in which the vocabulary and the counts are listed")
	parser.add_argument ("--mode", required=False, choices={"temporal", "source"}, default="source", help="mode for creation of the data file (default: source)")
	parser.add_argument ("--epochs", required=False, default=10, type=int, help="number of temporal divisions (default: 10)")
	parser.add_argument ("--ignore-file", required=True, type=str, help="file lists the files that should be ignored")
	args = parser.parse_args ()
	return args

def is_valid_word (word):
	""" accepts a word only if contains all alphabetic characters 
		or hyphen """
	word = word.strip()
	return re.match(r'^[a-zA-Z][A-Za-z-]*$', word)

def timestamp2epoch (timestamp, stride):
	nEpochs = int(1/stride)
	ranges = [(i*stride, (i+1) * stride) for i in range (nEpochs)]
	for i, (start, end) in enumerate (ranges):
		if timestamp >= start and timestamp < end:
			return i
	return nEpochs - 1

def date_format (string, format="%Y%m%d"):
	if len (string) == 6:
		format = "%Y%m"
	return format

class Reader (object):
	def __init__ (self, \
				  from_dir, \
				  to_file, \
				  epochs=10, \
				  sort_in_time=True, \
				  excluded_sources=[], \
				  exclude_lists=[], \
				  relabel_sources={}, \
				  file_ext=".txt", \
				  min_src_docs=100,
				  years_range=(1827, 1865)):
		self.from_dir = from_dir
		self.to_file = to_file
		self.epochs = epochs
		self.sort_in_time = sort_in_time
		self.excluded_sources = excluded_sources
		self.exclude_lists = exclude_lists
		self.relabel_sources = relabel_sources
		self.file_ext = file_ext
		self.min_src_docs = min_src_docs
		self.years_range = years_range

	def _walk (self, verbose=True):
		for dir_path, dir_names, files in os.walk (self.from_dir):
			if len (dir_names) == 0:
				source = dir_path.split("/")[-2]
				if source not in self.excluded_sources:
					date = dir_path.split ("/")[-1]
					for filename in files:
						if filename.endswith (self.file_ext):
							yield os.path.join (dir_path, filename), source, date

	def transfer (self, format="%Y%m%d", mode="source", verbose=False):
		# Read all the documents in memory.
		documents = [(filename, source, datetime.datetime.strptime(date, date_format (date, format))) for filename, source, date in self._walk (verbose=verbose)]
		excluded_files = set ()
		for filename in self.exclude_lists:
			with open (filename) as fin:
				for line in fin:
					excluded_files.add (line.strip())
            
		documents = [(filename, source, date) for filename, source, date in documents if filename not in excluded_files]

		# Relabel the sources (if necessary)
		if len (self.relabel_sources) > 0:
			documents = [(filename, self.relabel_sources.get (source, source), date) for filename, source, date in documents]

		all_sources = set ([source for _, source, _ in documents])
		all_time_bins = [f"T{i}" for i in range (self.epochs)]
		self.features = all_time_bins
		if mode == "source":
			self.features = self.features + [f"{bin_name}_{s}" for bin_name in all_time_bins for s in all_sources]
		

		self.min_date = min (documents, key=lambda x:x[2])[2]
		self.max_date = max (documents, key=lambda x:x[2])[2]
		min_time = self.min_date.timestamp ()
		max_time = self.max_date.timestamp ()

		stride = 1 / self.epochs

		if self.sort_in_time:
			ordered_collection = sorted (documents, key=lambda x:x[2])
		else:
			ordered_collection = documents

		records = 0
		self.vocab = defaultdict (int)
		with safe_open_w (self.to_file) as fout:
			for filename, source, date in ordered_collection:
				t = date.timestamp ()
				timestamp = (t - min_time)/(max_time-min_time)
				with open (filename) as fin:
					js = ujson.loads (fin.read().strip())
					text = js["corrected_text"]
					tokens = [token.lower() for token in text.split () if is_valid_word (token)]
					if len (tokens) == 0:
						continue
					for token in tokens:
						self.vocab[token] += 1
					tokenized_text = " ".join(tokens)
				bin_name=f"T{timestamp2epoch (timestamp, stride)}"
				source_name = f"{bin_name}_{source}"
				if mode == "source":
					fout.write ("\t".join ([filename, bin_name, source_name, tokenized_text]) + "\n")
				else:
					fout.write ("\t".join ([filename, bin_name, bin_name, tokenized_text]) + "\n")	
				records += 1

		if verbose:
			print (f"{self.to_file} contains {records} records")

	def dumpVocab (self, filename, sep="\t"):
		with safe_open_w (filename) as fout:
			for item in sorted (self.vocab.items(), key=lambda x:x[1], reverse=True):
				fout.write (f"{item[1]}{sep}{item[0]}\n")

	def dumpFeatures (self, filename):
		with safe_open_w (filename) as fout:
			for item in self.features:
				fout.write (f"{item}\n")

def main (args):
	relabel_sources = {"DouglassMonthly": "DouglassPapers", \
					   "FrederickDouglassPaper": "DouglassPapers", \
					   "TheNorthStar": "DouglassPapers", \
					   "TheColoredAmerican": "TheColoredAmerican", \
					   "WeeklyAdvocate": "TheColoredAmerican"}

	reader = Reader (args.data_dir, \
					 args.data_file, \
					 epochs=args.epochs, \
					 sort_in_time=True, \
					 #excluded_sources=["DouglassMonthly", "GodeysLadysBook", "TheCharlestonMercury-incomplete", "WeeklyAdvocate"], \
					 excluded_sources=["TheCharlestonMercury-incomplete"], \
					 exclude_lists=[args.ignore_file], \
					 relabel_sources=relabel_sources, \
					 file_ext=".json", \
					 years_range=(1827, 1865), \
					 min_src_docs=500
					 )
	reader.transfer (mode=args.mode, verbose=True)
	reader.dumpVocab (args.vocab_file, sep="\t")
	reader.dumpFeatures (args.features_file)

if __name__ == "__main__":
	main (readArgs ())
