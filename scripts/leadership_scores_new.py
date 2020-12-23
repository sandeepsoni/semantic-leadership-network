import argparse
import pandas as pd
import os
import sys
import numpy as np

if "../scripts" not in sys.path: sys.path.append ("../scripts")
from helpful_functions import readEmbeddings, normalize, sigmoid, logsigmoid, MAIN_FEAT

def readArgs ():
	parser = argparse.ArgumentParser (description="Comparing per word leadership statistic with a randomized dataset")
	parser.add_argument ("--rand-path", type=str, required=True, help="directory contains source conditional embeddings for randomized datasets")
	parser.add_argument ("--obs-path", type=str, required=True, help="directory that contains the source conditional embeddings for observed data")
	parser.add_argument ("--embeddings-file", type=str, required=True, help="embeddings file")
	parser.add_argument ("--feats-file", type=str, required=True, help="features file")
	parser.add_argument ("--leaders-file", type=str, required=True, help="leaders file")
	parser.add_argument ("--output-file", type=str, required=True, help="output file")
	args = parser.parse_args ()
	return args

def readDFAsDict (df, columns=["word", "s1", "s2", "t", "t+1", "Lead1"]):
	words = dict ()
	cols = columns[1:]
	for i, row in df[columns].iterrows():
		words[row["word"]] = [row[col] for col in cols]
	return words

def readFeats (filename):
	facet_names = [(MAIN_FEAT,)]
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split("_")
			if len (parts) > 1:
				facet_names.append ((parts[0], line.strip()))

	return facet_names

def transform_to_numpy (dict_embeddings, w2i, i2w, activated_facets, apply_normalization=False):
	if len(activated_facets) == 1 and MAIN_FEAT in activated_facets:
		mat = np.array([dict_embeddings[MAIN_FEAT][w] for i, w in enumerate (w2i)])
	else:
		activated_facets.insert (0, MAIN_FEAT)
		mat = np.array ([np.sum([dict_embeddings[facet_name][w] \
                                 for facet_name in activated_facets], axis=0) \
                         for i, w in enumerate (w2i)])
	if apply_normalization:
		mat = normalize (mat)
	return mat

def get_conditional_embeddings (deviations, w2i, i2w, facet_names, apply_normalization=False):
	# Calculate the true embeddings based on the deviations.
	fully_conditional_embeddings = dict ()
	for facet in facet_names:
		mat = transform_to_numpy (deviations, w2i, i2w, list(facet), apply_normalization=apply_normalization)
		if len (facet) == 1 and MAIN_FEAT in facet:
			fully_conditional_embeddings[MAIN_FEAT] = mat
		else:
			fully_conditional_embeddings[tuple(facet[1].split("_"))] = mat
	return fully_conditional_embeddings

def get_lead_score (w, deviations, embeddings, observed_stats, w2i, lead_type="l1"):
	_, s1, s2, t1, t2, lead1 = tuple (observed_stats)
	t1,t2 =  int(t1[1:]), int (t2[1:])
	lead = lead_measurement (deviations, embeddings, w, s1, s2, t1, t2, w2i, lead_type)
	return lead

def lead_measurement (deviations, embeddings, w, s1, s2, t1, t2, w2i, lead_type):
	# No point in calculating the lead-lag between sources
	# that were inactive at t1 and t2.
	# The following condition checks for that by measuring
	# a source-specific deviation at the given times.
	#if w not in deviations:
	#   return 1 # This shouldn't really happen but I will debug this later.
	if w not in deviations[f"T{t1}_{s1}"] or \
	   w not in deviations[f"T{t2}_{s2}"] or \
	   w not in deviations[f"T{t1}_{s2}"]:
		return -1 # word not present
    
	elif np.linalg.norm (deviations[f"T{t1}_{s1}"][w]) == 0 or \
		 np.linalg.norm (deviations[f"T{t2}_{s2}"][w]) == 0 or \
		 np.linalg.norm (deviations[f"T{t1}_{s2}"][w]) == 0:
		return -np.inf

	e1 = embeddings[(f"T{t1}",s1)][w2i[w]]
	e2 = embeddings[(f"T{t2}",s2)][w2i[w]]
	e3 = embeddings[(f"T{t1}",s2)][w2i[w]]

	num = np.dot (e1, e2)
	den = np.dot (e3, e2)

	if lead_type == "l1":
		return num/den

def main (args):
	# load the observed leadership data
	observed_df = pd.read_csv (os.path.join (args.obs_path, args.leaders_file), sep=";")	
	
	# Select word, s1, s2, t, t+1, lead1
	observed_dict = readDFAsDict(observed_df, columns=["word", "rank", "s1", "s2", "t", "t+1", "Lead1"])

	# Now go over the randomization run
	embeddings_file = os.path.join (args.rand_path, args.embeddings_file)
	embeddings = readEmbeddings (embeddings_file)
	static_embeddings = embeddings[MAIN_FEAT]

	# vocabulary
	w2i = {w:i for i,w in enumerate (static_embeddings)}
	i2w = {i:w for i,w in enumerate (static_embeddings)}
	vocab_size = len (w2i)

	facets_file = os.path.join (args.rand_path, args.feats_file)
	facet_names = readFeats (facets_file)
    
	conditional_embeddings = get_conditional_embeddings (embeddings, w2i, i2w, facet_names)

	randomized_dict = dict ()
	for w in observed_dict:
		lead = get_lead_score (w, embeddings, conditional_embeddings, observed_dict[w], w2i, "l1")
		randomized_dict[w] = observed_dict[w] + [lead]

	# write to file
	with open (os.path.join (args.rand_path, args.output_file), "w") as fout:
		header = f"word,rank,s1,s2,t,t+1,lead1,random_lead1"
		fout.write (f"{header}\n")
		for key, value in sorted (randomized_dict.items(), key=lambda x:x[1][0], reverse=False):
			text = f"{key},{value[0]},{value[1]},{value[2]},{value[3]},{value[4]},{value[5]:.4f},{value[6]:.4f}"
			fout.write (f"{text}\n")

if __name__ == "__main__":
	main (readArgs ())
