"""
python name_identification.py --dir-path ../data/aa_temp.1/ --scores-file local.scores --embeddings-file out.embeddings --names-file names.ignore --casestats-file ../annotations/wordcase.stats --annotations-file ../annotations/names_locations.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from helpful_functions import readEmbeddings, normalize


def readArgs ():
	parser = argparse.ArgumentParser (description="classify words as either names or not")
	parser.add_argument ("--dir-path", required=True, type=str, help="path to the directory")
	parser.add_argument ("--scores-file", required=True, type=str, help="scores file")
	parser.add_argument ("--embeddings-file", required=True, type=str, help="embeddings file")
	parser.add_argument ("--casestats-file", required=True, type=str, help="wordcase statistics file")
	parser.add_argument ("--annotations-file", required=True, type=str, help="annotations file")
	parser.add_argument ("--names-file", required=True, type=str, help="names file")
	args = parser.parse_args ()
	return args

def load_dataset (propernames, non_propernames, emds, idx, iidx):
	name_indices = [idx[item] for item in propernames if item in idx]
	non_name_indices = [idx[item] for item in non_propernames if item in idx]
	X = emds[non_name_indices + name_indices]
	y = np.array([0 for item in non_name_indices] + [1 for item in name_indices])
	print (len(name_indices), len (non_name_indices))
	return X, y

def run_classifier (positive_examples, negative_examples, main_embeddings, w2i, i2w):
	X, y = load_dataset (positive_examples, negative_examples, main_embeddings, w2i, i2w)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

	clf = LogisticRegression(random_state=42).fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	target_names = ['Non-name', 'Name']
	print(classification_report(y_test, y_pred, target_names=target_names))
    
	# Apply to all the words in the vocabulary
	predictions = clf.predict (main_embeddings)
	return predictions

def get_all_names_and_locations (positive_annotations, negative_annotations, embeddings, w2i, i2w):
	predictions = run_classifier (positive_annotations, negative_annotations, embeddings, w2i, i2w)
	predicted_names_locations = set ()
	for i, item in enumerate (predictions):
		if item == 1:
			predicted_names_locations.add (i2w[i])

	return predicted_names_locations

def generate_annotations (filename, w2i):
	with open (filename) as fin:
		annotations = set ()
		for line in fin:
			annotations.add (line.strip())

	neg_annotations = random.choices ([w for w in w2i if w not in annotations], k=len (annotations))
	print (len(annotations), len (neg_annotations))

	return annotations, neg_annotations

def main (args):
	embeddings = readEmbeddings (os.path.join (args.dir_path, args.embeddings_file))
	static_embeddings = embeddings["MAIN"]

	w2i = {w:i for i,w in enumerate (static_embeddings)}
	i2w = {i:w for i,w in enumerate (static_embeddings)}

	# the atemporal embeddings
	main_embeddings = np.array([static_embeddings[i2w[i]] for i in range (len(i2w))])
	main_embeddings = normalize (main_embeddings)

	# read the words and their change scores
	ranks = pd.read_csv (os.path.join (args.dir_path, args.scores_file), sep=";")

	# names and locations identification using name embeddings
	positives, negatives = generate_annotations (args.annotations_file, w2i)

	names_locations_by_embeddings = get_all_names_and_locations (positives, negatives, main_embeddings, w2i, i2w)

	# names and locations identification using wordcase statistics	
	wordcase_stats = pd.read_csv (args.casestats_file, sep="\t")
	casestats_names_locations = set(list(wordcase_stats[(wordcase_stats.upperp > 0.8) | (wordcase_stats.titlep > 0.8) | (wordcase_stats.lowerp <= 0.2)].word.values))
	names_locations_by_casestats = {w for w in w2i if w in casestats_names_locations}
	names_locations = names_locations_by_embeddings & names_locations_by_casestats
	
	# write the names and locations to file
	with open (os.path.join (args.dir_path, args.names_file), "w") as fout:
		for w in names_locations:
			fout.write (f"{w}\n")

if __name__ == "__main__":
	main (readArgs ())
