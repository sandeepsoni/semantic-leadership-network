import argparse
import os
import numpy as np
import logging
from collections import defaultdict
import pickle
from helpful_functions import readEmbeddings, normalize
from mylib import semantic_neighbors

def readArgs ():
	parser = argparse.ArgumentParser (description="Near negihbors for words")
	parser.add_argument ("--dir-path", required=True, type=str, help="directory path")
	parser.add_argument ("--embeddings-file", required=True, type=str, help="embeddings file")
	parser.add_argument ("--near-neighbors-file", required=True, type=str, help="near neighbors file")
	parser.add_argument ("--facet-name", required=False, type=str, default="MAIN", help="name of the facet (default: MAIN)")
	parser.add_argument ("--nearest", required=False, type=int, default=25, help="number of near neighbors (default: 25)")
	args = parser.parse_args ()
	return args

def getNeighbors (all_embeddings, w2i, i2w, k=25, log_every=1000):
	neighbors = defaultdict (list)
	for index, w in enumerate (w2i):
		for i in range (len (all_embeddings)):
			neighbors[w].append (semantic_neighbors (w, all_embeddings[i], (w2i, i2w), k=k))

		if (index+1) % log_every == 0:
			logging.info (f"Words processed: {index+1}, Percentage: {(index+1)/len(w2i)}")            
    
	return neighbors

def main (args):
	embeddings = readEmbeddings (os.path.join (args.dir_path, args.embeddings_file))

	# Separate the main embeddings and the facet embeddings
	static_embeddings = embeddings["MAIN"]
	
	# vocabulary
	w2i = {w:i for i, w in enumerate (static_embeddings)}
	i2w = {i:w for i, w in enumerate (static_embeddings)}

	# the atemporal embeddings
	main_embeddings = np.array([static_embeddings[i2w[i]] for i in range (len(i2w))])
	main_embeddings = normalize (main_embeddings)

	if not args.facet_name == "MAIN":
		residual_embeddings = embeddings[args.facet_name]
		# temporal embeddings (just add the static embeddings to the facets)
		temporal_embeddings = normalize(np.array([static_embeddings[i2w[i]] + residual_embeddings[i2w[i]] \
                                                  for i in range (len(i2w))]))


	# all the embeddings (the atemporal embeddings are first followed by all the temporal embeddings)
	all_embeddings = list ()
	if args.facet_name == "MAIN":
		all_embeddings.append (main_embeddings)
	else:
		all_embeddings.append (temporal_embeddings)

	neighbors = getNeighbors (all_embeddings, w2i, i2w, k=args.nearest)

	# write the neighbors to file
	with open (os.path.join (args.dir_path, args.near_neighbors_file), "wb") as fout:
		pickle.dump (neighbors, fout)

if __name__ == "__main__":
	main (readArgs ())
