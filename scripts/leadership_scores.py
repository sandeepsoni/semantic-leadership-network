import argparse
import pandas as pd
import os
import itertools
import math
import numpy as np
import logging
logging.basicConfig (format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
from helpful_functions import readEmbeddings, normalize, sigmoid, logsigmoid, MAIN_FEAT

def readArgs ():
	parser = argparse.ArgumentParser (description="leadership score calculation")
	parser.add_argument ("--src-path", type=str, required=True, help="directory that contains the source embeddings")
	parser.add_argument ("--temp-path", type=str, required=True, help="directory that contains the temporal embeddings and other files")
	parser.add_argument ("--embeddings-file", type=str, required=True, help="embeddings file")
	parser.add_argument ("--feats-file", type=str, required=True, help="listing the features in a file")
	parser.add_argument ("--changes-file", type=str, required=True, help="file contains all the changes")
	parser.add_argument ("--leaders-file", type=str, required=True, help="file that contains the leadership scores")
	parser.add_argument ("--lead-types", type=str, nargs="+", required=True, help="short codes for lead types")
	args = parser.parse_args ()
	return args

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

def get_leader_dyads (words, deviations, embeddings, sources, w2i, lead_type):
	instance_specific_leads = dict ()
	for i, item in enumerate (words):
		w,t1,t2,f1,f2 = str(item[0]), int(item[1][1:]), int(item[2][1:]), item[3], item[4]
		if w in w2i:
			leads = {(s1, s2, x, y): lead_measurement (deviations, embeddings, w, s1, s2, x, y, f1, f2, w2i, lead_type) \
									 for s1, s2 in itertools.permutations (sources, 2) \
									 for x,y in zip (range (t1, t2), range (t1+1, t2+1))}

			leads = {key: leads[key] for key in leads if leads[key] != -np.inf} # keep only those pairs which have leads
			if len (leads) > 0:
				key, value = max (leads.items(), key=lambda x:x[1])
			else:
				key, value = None, None
			instance_specific_leads[w] = (key, value)
	return instance_specific_leads

def lead_measurement (deviations, embeddings, w, s1, s2, t1, t2, f1, f2, w2i, lead_type):
	# No point in calculating the lead-lag between sources 
	# that were inactive at t1 and t2.
	# The following condition checks for that by measuring 
	# a source-specific deviation at the given times.
	#if w not in deviations:
	#	return 1 # This shouldn't really happen but I will debug this later.

	V = len (w2i)
	k = 50 # I would like to remove this hardcoded code
	const = math.log (V/k) 

	if np.linalg.norm (deviations[f"T{t1}_{s1}"][w]) == 0 or \
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
	elif lead_type == "l2":
		return (f2 * (logsigmoid (num) - logsigmoid(den)))
	elif lead_type == "l3":
		return (f2 * (logsigmoid (num + const) - logsigmoid (den + const)))
	elif lead_type == "l4":
		return logsigmoid (num) - logsigmoid(den)
	elif lead_type == "l5":
		return logsigmoid (num + const) - logsigmoid (den + const)	
	
def get_leader_words (dictionary):
	return [key for key in dictionary if not dictionary[key] == (None,None)]

def writeToFile (filename, changes, sep=";"):
	changes.to_csv (filename, sep=sep, header=True, index=False)	

def add2df (changes, dyads, lead_type):
	"""
	dyads is a dictionary with word as the key.
	"""
	lead_types = {"l1": "Lead1", "l2": "Lead2", "l3": "Lead3", "l4": "Lead4", "l5": "Lead5"}

	rows = list ()
	header = changes.columns.values.tolist()
	for index, row in changes.iterrows ():
		w = str(row["word"])
		if w in dyads:
			new_row = [row[name] for name in header]
			key, value = dyads[w]
			if key is not None and value is not None:
				s1,s2,x,y = key
				lead = value
				new_row.extend ([s1, s2, f"T{x}", f"T{y}", lead])
				rows.append (new_row)

	new_df = pd.DataFrame (rows, columns=header + [f"{lead_types[lead_type]}_s1", f"{lead_types[lead_type]}_s2", f"{lead_types[lead_type]}_t", f"{lead_types[lead_type]}_t+1", lead_types[lead_type]])
	return new_df

def main (args):
	embeddings_file = os.path.join (args.src_path, args.embeddings_file)
	embeddings = readEmbeddings (embeddings_file)
	static_embeddings = embeddings[MAIN_FEAT]
	logging.info (f"Embeddings read from {embeddings_file}")

	# vocabulary
	w2i = {w:i for i,w in enumerate (static_embeddings)}
	i2w = {i:w for i,w in enumerate (static_embeddings)}
	vocab_size = len (w2i)
	logging.info (f"Vocabulary mapping done, total words {len(w2i)}")
	
	facets_file = os.path.join (args.src_path, args.feats_file)
	facet_names = readFeats (facets_file)
	logging.info (f"Read {len(facet_names)} facets from {facets_file}")

	changes_file = os.path.join (args.temp_path, args.changes_file)
	candidates = pd.read_csv (changes_file, sep=";")
	all_words = candidates[["word", "Period1", "Period2", "Freq1", "Freq2"]].values.tolist()
	sources = set ([facet[1].split("_")[1] for facet in facet_names if len (facet) > 1])

	for lead_type in args.lead_types:
		if lead_type != "l1":
			conditional_embeddings = get_conditional_embeddings (embeddings, w2i, i2w, facet_names, apply_normalization=True)
		else:
			conditional_embeddings = get_conditional_embeddings (embeddings, w2i, i2w, facet_names)
			
		leader_dyads = get_leader_dyads (all_words, embeddings, conditional_embeddings, sources, w2i, lead_type) 
		candidates = add2df (candidates, leader_dyads, lead_type)
		logging.info (f"Leadership scores calculated for {len(leader_dyads)} for lead type {lead_type}")

	# write to file
	leaders_file = os.path.join (args.src_path, args.leaders_file)
	writeToFile (leaders_file, candidates)
	logging.info (f"All leader values written to {leaders_file}")
	
if __name__ == "__main__":
	main (readArgs ())
