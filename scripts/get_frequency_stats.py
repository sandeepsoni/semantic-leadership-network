import argparse
import os
from collections import defaultdict
from helpful_functions import init_dict, source_names, name_changes

def readArgs ():
	parser = argparse.ArgumentParser (description="get the period and source specific frequencies")
	parser.add_argument ("--dir-path", required=True, type=str, help="directory that contains the files")
	parser.add_argument ("--data-file", required=True, type=str, help="file that contains all the data")
	parser.add_argument ("--word-freq-file", required=True, type=str, help="file that contains the word frequency")
	parser.add_argument ("--doc-freq-file", required=True, type=str, help="file that contains the document frequency")
	parser.add_argument ("--period-freq-file", required=True, type=str, help="file that contains the period frequency")
	parser.add_argument ("--source-freq-file", required=True, type=str, help="file that contains the source frequency")
	parser.add_argument ("--periods", required=False, type=int, default=10, help="specify the number of periods")
	args = parser.parse_args ()
	return args

def main (args):
	# For each word in the vocab, calculate the word frequency and the document frequency. 
	# We want to pick words that have either high token frequencies or document frequencies.
	# For each word in the vocab, calculate the frequency of the word in each period, and in each source.
	word_freq_vocab = defaultdict (int)
	doc_freq_vocab = defaultdict (int)

	period_freq_vocab = dict ()
	source_freq_vocab = dict ()

	period_names = [f'T{i}' for i in range (args.periods)]

	with open (os.path.join (args.dir_path, args.data_file)) as fin:
		for i, line in enumerate (fin):
			parts = line.strip().split("\t")
			source = parts[0].strip("/").split("/")[3]
			if source in name_changes:
				source = name_changes[source]
			period = parts[1]
			period = parts[1]
			text = parts[3]
        
			for w in text.split():
				word_freq_vocab[w] += 1
        
			for w in set (text.split()):
				doc_freq_vocab[w] += 1
            
			for w in text.split ():
				if w not in period_freq_vocab:
					period_freq_vocab[w] = init_dict (period_names)
				period_freq_vocab[w][period] += 1
            
				if w not in source_freq_vocab:
					source_freq_vocab[w] = init_dict (source_names)
				source_freq_vocab[w][source] += 1
            
			if (i+1) % 50000 == 0:
				print (f"{i+1} documents processed")
                
	print (f"{i+1} documents processed overall")

	with open (os.path.join (args.dir_path, args.word_freq_file), "w") as fout:
		for word in word_freq_vocab:
			fout.write (f"{word},{word_freq_vocab[word]}\n")

	with open (os.path.join (args.dir_path, args.doc_freq_file), "w") as fout:
		for word in doc_freq_vocab:
			fout.write (f"{word},{doc_freq_vocab[word]}\n")

	with open (os.path.join (args.dir_path, args.period_freq_file), "w") as fout:
		for word in period_freq_vocab:
			for period in period_freq_vocab[word]:
				fout.write (f"{word},{period},{period_freq_vocab[word][period]}\n")

	with open (os.path.join (args.dir_path, args.source_freq_file), "w") as fout:
		for word in source_freq_vocab:
			for source in source_freq_vocab[word]:
				fout.write (f"{word},{source},{source_freq_vocab[word][source]}\n")

if __name__ == "__main__":
	main (readArgs ())
