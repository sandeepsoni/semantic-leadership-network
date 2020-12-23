import numpy as np

def semantic_neighbors(word:str, embs:np.array, voc:tuple, k=3) -> list:
	""" Get the list of near neighbors for a given word from the embeddings.

		Each row of the matrix `embs` is a vector for a word.
		The mapping of words and row numbers is in `voc`.

		NOTE: Assumes that the embeddings are unit vectors.
	"""
	w2i, i2w = voc
	sims = np.dot(embs[w2i[word],], embs.T)
	
	output = []
	for sim_idx in sims.argsort()[::-1][1:(1+k)]:
		if sims[sim_idx] > 0:
			output.append((sims[sim_idx], i2w[sim_idx]))
	return output

