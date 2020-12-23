"""
python local_dynamic_ranks.py --dir-path ../data/aa_temp.1 --embeddings-file out.embeddings --scores-file local.scores
"""

import argparse
import numpy as np
from collections import defaultdict
import pickle
import os
import itertools
import pandas as pd
from helpful_functions import readEmbeddings, normalize

def readArgs ():
	parser = argparse.ArgumentParser (description="Top changed words")
	parser.add_argument ("--dir-path", required=True, type=str, help="directory that contains the required files")
	parser.add_argument ("--embeddings-file", required=True, type=str, help="filename of the embeddings")
	parser.add_argument ("--k", required=False, type=int, default=10, help="number of near neighbors to be used (default: 10)")
	parser.add_argument ("--scores-file", required=True, type=str, help="file contains the ranked list of words based on quantity of semantic change")
	args = parser.parse_args ()
	return args

def emds2temporal (embeddings, facet_names):
	# Separate the main embeddings and the facet embeddings
	static_embeddings = embeddings["MAIN"]
	
	# vocabulary
	w2i = {w:i for i, w in enumerate (static_embeddings)}
	i2w = {i:w for i, w in enumerate (static_embeddings)}

	# the atemporal embeddings
	main_embeddings = np.array([static_embeddings[i2w[i]] for i in range (len(i2w))])
	main_embeddings = normalize (main_embeddings)
	
	residual_embeddings = {facet_name: embeddings[facet_name] for facet_name in facet_names}
	temporal_embeddings = {facet_name: normalize(np.array([static_embeddings[i2w[i]] + residual_embeddings[facet_name][i2w[i]] for i in range (len(i2w))])) for facet_name in facet_names}


	return main_embeddings, temporal_embeddings, (w2i, i2w)

def readNeighbors (files):
	neighbors = dict ()
	for filename in files:
		basename = os.path.basename (filename)
		facet = basename.split (".")[0]
		with open (filename, "rb") as fin:
			neighbor = pickle.load (fin)
			neighbors[facet] = neighbor
	
	return neighbors

def cos_dist (vec1, vec2):
	l1 = np.linalg.norm (vec1)
	l2 = np.linalg.norm (vec2)

	sim = (np.dot (vec1, vec2)/(l1*l2))
	return 1-sim	

def getScores (embeddings, neighbors, voc, k=10):
	scores = defaultdict (list)
	w2i, i2w = voc
	facets = [key for key in embeddings]
	for word in w2i:
		for f1, f2 in itertools.combinations (facets, 2):
			neighbors1 = neighbors[f1][word][0]
			neighbors2 = neighbors[f2][word][0]
			s1, n1 = {n:s for s,n in neighbors1}, [n for _,n in neighbors1]
			s2, n2 = {n:s for s,n in neighbors2}, [n for _,n in neighbors2]
			common = set (n1[0:k]).union (n2[0:k])
			s1_vec = [np.dot (embeddings[f1][w2i[word]], embeddings[f1][w2i[n]]) for n in common]
			s2_vec = [np.dot (embeddings[f2][w2i[word]], embeddings[f2][w2i[n]]) for n in common]
			dist = cos_dist (s1_vec, s2_vec)
			scores[word].append ((f1, f2, dist))
			
	return scores

def getMaximalScores (scores, ignore="MAIN"):
	maximal = dict ()
	for w in scores:
		score = [item for item in scores[w] if not item[0] == ignore and not item[1] == ignore]
		maximal[w] = max (score, key=lambda x:x[2])
	return maximal

def getRankedList (maximal_scores):
	ranks = {word: i for i, (word, item) in enumerate (sorted (maximal_scores.items(), key=lambda x:x[1][2], reverse=True))}
	return ranks

def writeResults (scores, ranks, neighbors, filename, k=10, sep=";"):	
	items = list ()
	for word, rank in sorted (ranks.items(), key=lambda x:x[1]):
		p1, p2, score = scores[word]
		n1 = [n for _,n in neighbors[p1][word][0][0:k]]
		n2 = [n for _,n in neighbors[p2][word][0][0:k]]

		items.append ([word, rank, p1, p2, n1, n2, score])

	df = pd.DataFrame (items, columns=["word", "rank", "Period1", "Period2", "Neighbors1", "Neighbors2", "Score"])
	df.to_csv (filename, sep=sep, header=True, index=False)

def main (args):	
	embeddings = readEmbeddings (os.path.join (args.dir_path, args.embeddings_file))
	facet_names = [key for key in embeddings if not key == "MAIN"]
	main_embeddings, temporal_embeddings, voc = emds2temporal(embeddings, facet_names)
	
	w2i, i2w = voc

	neighbors = readNeighbors ([os.path.join (args.dir_path, f"{facet_name}.neighbors.pkl") for facet_name in facet_names])
	scores = getScores (temporal_embeddings, neighbors, voc, k=args.k)
	maximal_scores = getMaximalScores (scores)
	ranks = getRankedList (maximal_scores)
	writeResults (maximal_scores, ranks, neighbors, os.path.join (args.dir_path, args.scores_file), k=10, sep=";")

if __name__ == "__main__":
	main (readArgs())
