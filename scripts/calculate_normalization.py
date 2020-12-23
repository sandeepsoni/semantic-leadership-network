import numpy as np
import os
from helpful_functions import readEmbeddings, normalize, MAIN_FEAT
import argparse
import logging
logging.basicConfig (format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

def readArgs ():
	parser = argparse.ArgumentParser (description="calculate the normalization part")
	parser.add_argument ("--src-dir", type=str, required=True, help="source directory")
	parser.add_argument ("--embeddings-file", type=str, required=True, help="embeddings file")
	parser.add_argument ("--contexts-file", type=str, required=True, help="contexts file")
	parser.add_argument ("--features-file", type=str, required=True, help="features file")
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

def get_conditional_embeddings (deviations, w2i, i2w, facet_names):
	# Calculate the true embeddings based on the deviations.
	fully_conditional_embeddings = dict ()
	for facet in facet_names:
		mat = transform_to_numpy (deviations, w2i, i2w, list(facet), apply_normalization=False)
		if len (facet) == 1 and MAIN_FEAT in facet:
			fully_conditional_embeddings[MAIN_FEAT] = mat
		else:
			fully_conditional_embeddings[tuple(facet[1].split("_"))] = mat
	return fully_conditional_embeddings

def transform_to_numpy (dict_embeddings, w2i, i2w, activated_facets, apply_normalization=True):
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

def readContextEmbeddings (filename):
	embeddings = dict ()
	with open (filename) as fin:
		for i, line in enumerate (fin):
			if i == 0:
				parts = line.strip().split ()
				vocab_size, dims = int(parts[0]), int (parts[1])
			else:
				parts = line.strip().split ()
				word = parts[0]
				values = np.array(list(map(float, parts[1:])))
				embeddings[word] = values
	return embeddings

def main (args):
	embeddings_file = os.path.join (args.src_dir, args.embeddings_file)
	embeddings = readEmbeddings (embeddings_file)
	static_embeddings = embeddings[MAIN_FEAT]
	logging.info (f"Embeddings read from {embeddings_file}")

	# vocabulary
	w2i = {w:i for i,w in enumerate (static_embeddings)}
	i2w = {i:w for i,w in enumerate (static_embeddings)}
	logging.info (f"Vocabulary mapping done, total words {len(w2i)}")

	features_file = os.path.join (args.src_dir, args.features_file)
	facet_names = readFeats (features_file)
	logging.info (f"Read {len(facet_names)} facets from {features_file}")
	
	facets = [f[1] for f in facet_names if len (f) > 1]
	idx_facets = {facet: i for i,facet in enumerate (facets)}
	iidx_facets = {i: facet for i,facet in enumerate (facets)}

	contexts_file = os.path.join (args.src_dir, args.contexts_file)
	conditional_embeddings = get_conditional_embeddings (embeddings, w2i, i2w, facet_names)
	context_embeddings = readContextEmbeddings (contexts_file)
	context_embeddings = np.array([context_embeddings[w] for i, w in enumerate (w2i)])
	print (context_embeddings.nbytes)
	context_embeddings = context_embeddings.astype (np.float16)
	print (context_embeddings.nbytes)
	return

	logging.info (f"Read context embeddings from {contexts_file}")

	Z = list ()
	for i,facet in sorted (iidx_facets.items(), key=lambda x:x[0]):
		key = tuple (facet.split("_"))
		C = np.dot (context_embeddings, conditional_embeddings[key].T)
		z = np.log (np.sum(np.exp(C), axis=0))
		Z.append (z)
		if (i+1) % 10 == 0:
			logging.info (f"normmalization at {i+1} facet")
	Z = np.array (Z)

	normalization_file = os.path.join (args.src_dir, "Z.npy")
	with open (normalization_file , "wb") as f:
		np.save(f, Z)

	logging.info (f"Normalization values stored in {normalization_file}")

	facets_file = os.path.join (args.src_dir, "facets.txt")

	with open (facets_file, "w") as fout:
		for i, facet in sorted (iidx_facets.items(), key=lambda x:x[0]):
			fout.write (f"{facet}\n")

	logging.info (f"Facet index stored in  {facets_file}")
	
if __name__ == "__main__":
	main (readArgs ())
