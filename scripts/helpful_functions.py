import os
import re
import math
import errno
import numpy as np

def walk (from_dir, prefix="", file_ext=".html"):
	""" Iterator over files in a directory. """
	for dir_path, dir_names, files in os.walk (from_dir):
		if len (dir_names) == 0:
			for filename in files:
				if filename.endswith (file_ext) and filename.startswith (prefix):
					yield os.path.join (dir_path, filename)	

def basic_preprocess (text):
	text = text.replace (",", " ")
	text = text.replace (".", " ")
	text = text.replace ("'", "")
	text = text.replace ('"', "")
	text = text.replace (";", " ")
	text = text.replace (":", " ")
	text = text.replace ("!", " ")
	text = text.replace ("?", " ")
	#text = text.replace ("-", " ")
	text = text.replace ("[", " ")
	text = text.replace ("]", " ")
	text = text.replace ("{", " ")
	text = text.replace ("}", " ")

	# remove em-dashes
	text = re.sub (r'\u2014', ' ', text)
	# remove quotations
	text = re.sub (r'\u201D', '', text)
	text = re.sub (r'\u201C', '', text)
	# replace $ amounts by special token
	text = re.sub (r"\$\d+(?:\.\d+)?", "<currency>", text)
	# replace numbers by special token 
	text = re.sub(r"^(\d+)$",'<number>', text)
	return text

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
	try:
		os.makedirs(path, exist_ok=True)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

# Taken from https://stackoverflow.com/a/23794010
def safe_open_w(path):
	""" Open "path" for writing, creating any parent directories as needed.
	"""
	mkdir_p(os.path.dirname(path))
	return open(path, 'w')

## functions that are specific to reading and processing embeddings

def readEmbeddings (filename):
	embeddings = dict ()
	with open (filename) as fin:
		for i, line in enumerate (fin):
			if i == 0:
				parts = line.strip().split ()
				vocab_size, dims = int(parts[0]), int (parts[1])
			else:
				parts = line.strip().split ()
				facet_name = parts[0]
				word = parts[1]
				values = np.array(list(map(float, parts[2:])))
				if facet_name not in embeddings:
					embeddings[facet_name] = dict ()
				embeddings[facet_name][word] = values
	return embeddings

def normalize(embeddings):
	norms = np.linalg.norm(embeddings, ord=2, axis=1)
	new_matrix = embeddings / norms[:, np.newaxis]
	return new_matrix

def sigmoid (x):
	return 1/(1+math.exp(-x))

def logsigmoid (x):
	return math.log (sigmoid (x))

def is_count_greater (values, thresh=5):
	return sum(values) >= thresh

def is_count_smaller (values, thresh=5):
	return sum(values) < thresh

source_names = ["DouglassPapers", \
				"FrankLesliesWeekly", \
				"FreedomsJournal", \
				"GodeysLadysBook", \
				"NationalAntiSlaveryStandard", \
				"ProvincialFreeman", \
				"TheChristianRecorder", \
				"TheColoredAmerican", \
				"TheLiberator", \
				"TheLily", \
				"TheNationalEra"]


name_changes = {"WeeklyAdvocate": "TheColoredAmerican", \
				"TheNorthStar": "DouglassPapers", \
				"FrederickDouglassPaper": "DouglassPapers", \
				"DouglassMonthly": "DouglassPapers"}

def init_dict (names):
    return {name: 0 for name in names}

START_TIME=1827
END_TIME=1865

#N_EPOCHS = len (period_names)

#STRIDE = (END_TIME - START_TIME) / N_EPOCHS

MAIN_FEAT="MAIN"
