import argparse
import pandas as pd
import os
import random
from random import choices, sample
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from helpful_functions import safe_open_w

def readArgs ():
	parser = argparse.ArgumentParser (description="swapping code ")
	parser.add_argument ("--src-file", type=str, required=True, help="file contains all the observed data")
	parser.add_argument ("--tgt-file", type=str, required=True, help="file contains all the randomly permuted data")
	parser.add_argument ("--chunk-size", type=int, required=False, default=1000, help="size of each document")
	parser.add_argument ("--max-source-size", type=int, required=False, default=None, help="total overall tokens for a source-time combination")
	parser.add_argument("--keep-all", dest="keep_all", action="store_true")
	parser.add_argument("--no-keep-all", dest="keep_all", action="store_false")
	parser.set_defaults(keep_all=True)

	parser.add_argument ("--epochs", type=str, nargs="+", required=False, default=[], help="the epochs that need to be kept")

	parser.add_argument("--always-activated", dest="always_activated", action="store_true")
	parser.add_argument("--not-always-activated", dest="always_activated", action="store_false")
	parser.set_defaults(always_activated=False)


	args = parser.parse_args ()
	if not args.keep_all and len(args.epochs) == 0:
		parser.error('must have non-zero number --epochs when --no-keep-all')
	return args

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def make_chunks (text, chunk_size):
	return [chunk for chunk in chunks (text, chunk_size)]

def read_data (filename, chunk_size=1000):
	# read the data as it is
	docs = list ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split ("\t")
			epoch = parts[1]
			source = parts[2].split ("_")[1]
			text = parts[3]
			docs.append ([epoch, source, text])

	# group all documents from one epoch and 
	# one source
	grouped = defaultdict (list)
	for doc in docs:
		epoch, src, text = doc[0], doc[1], doc[2]
		grouped[(epoch, src)].append (doc[2])

	# coalesce all documents into one
	grouped = {key: [token for text in grouped[key] for token in text.split()] for key in grouped}

	# and then make document chunks
	grouped = {key: make_chunks(grouped[key], chunk_size) for key in grouped}
	
	rows = list ()
	for key in grouped:
		epoch, src = key
		for chunk in grouped[key]:
			rows.append ([epoch, src, chunk])
	
	df = pd.DataFrame (rows, columns=["epoch", "orig_source", "text"])
	return df

def transform_by_permuting (df):
	# add modified_src column
	df["mod_source"] = df["orig_source"]
	
	# Per epoch, print the initial source distribution
	epochs = df["epoch"].unique()
	src_dist = dict ()
	sources = list ()
	for epoch in epochs:
		epoch_sources = df[df["epoch"] == epoch]["mod_source"].values
		new_sources = np.random.permutation (epoch_sources)
		sources.append (new_sources)
		
	# assign the new sources
	df["mod_source"] = pd.Series (np.concatenate (sources, axis=0))
	return df


def select_docs (df, max_docs=None):
	if max_docs is None:
		return df

	# reorient into a epoch-source dictionary
	data = dict ()
	for index, row in df.iterrows():
		epoch, mod_source = row["epoch"], row["mod_source"]
		if epoch not in data:
			data[epoch] = dict ()
		if mod_source not in data[epoch]:
			data[epoch][mod_source] = list ()

		data[epoch][mod_source].append ((row["orig_source"], row["text"]))
		
		
	# now sweep over every epoch one at a time; then every source one at a time;
	# and then select a sample for each combination 
	modified_rows = list ()
	for epoch in data:
		for source in data[epoch]:
			if len (data[epoch][source]) <= max_docs:
				# just copy everything
				for item in data[epoch][source]:
					modified_rows.append ([epoch, source, item[0], item[1]])
			else:
				items = sample (data[epoch][source], max_docs)
				for item in items:
					modified_rows.append ([epoch, source, item[0], item[1]])

	
	mod_df = pd.DataFrame (modified_rows, columns=["epoch", "mod_source", "orig_source", "text"])
	return mod_df

def select_based_on_time (df, epochs, activated_throughout=False):	
	# reorient into an epoch-source dictionary
	data = dict ()
	for index, row in df.iterrows():
		epoch, mod_source = row["epoch"], row["mod_source"]
		if epoch not in data:
			data[epoch] = dict ()
		if mod_source not in data[epoch]:
			data[epoch][mod_source] = list ()

		data[epoch][mod_source].append ((row["orig_source"], row["text"]))

	# now select only the relevant source-epoch pairs
	if len(epochs) > 0 and activated_throughout:
		# there are a few selected epochs and we want all sources 
		# that are activated throughout them
		source_map = dict ()
		for epoch in data:
			if epoch in epochs:
				for source in data[epoch]:
					if source not in source_map:
						source_map[source] = list ()
					source_map[source].append (epoch)

		relevant_sources = {key for key in source_map if len(source_map[key]) == len (epochs)}

	elif len (epochs) > 0:
		# there are few selected epochs and we select all sources within those epochs
		relevant_sources = set ()
		for epoch in data:
			if epoch in epochs:
				for source in data[epoch]:
					relevant_sources.add (source)

	elif activated_throughout:
		# there are not selected epochs but we only want those sources that are activated throughout
		source_map = dict ()
		for epoch in data:
			epochs.append (epoch)
			for source in data[epoch]:
				if source not in source_map:
					source_map[source] = list ()
				source_map[source].append (epoch)

		relevant_sources = {key for key in source_map if len(source_map[key]) == len (epochs)}
	else:
		# select everything
		relevant_sources = set ()
		for epoch in data:
			epochs.append (epoch)
			for source in data[epoch]:
				relevant_sources.add (source)

	# now create a dataframe out of it
	modified_rows = list ()
	for epoch in data:
		for source in data[epoch]:
			if epoch in epochs and source in relevant_sources:
				for item in data[epoch][source]:
					modified_rows.append ([epoch, source, item[0], item[1]])
		
	mod_df = pd.DataFrame (modified_rows, columns=["epoch", "mod_source", "orig_source", "text"])
	return mod_df

def write_data (df, filename):
	with safe_open_w (filename) as fout:
		for index, row in df.iterrows():
			tokens = " ".join (row["text"])
			epoch = row["epoch"]
			src = row["mod_source"]
			orig = row["orig_source"]
			intersection = f"{epoch}_{src}"
			fout.write (f"{orig}\t{epoch}\t{intersection}\t{tokens}\n")

def main (args):
	# load data and create per source document chunks
	df = read_data (args.src_file, chunk_size=args.chunk_size)
	new_df = transform_by_permuting (df)
	
	# restrict to max number of documents per source in every epoch
	new_df = select_docs (new_df, max_docs = int(args.max_source_size/args.chunk_size))

	# restrict further to small number of epochs if necessary
	# and decide which source time pairs are to be kept
	if not args.keep_all:
		new_df = select_based_on_time (new_df, args.epochs, args.always_activated)
	write_data (new_df, args.tgt_file)
	
if __name__ == "__main__":
	main (readArgs ())
