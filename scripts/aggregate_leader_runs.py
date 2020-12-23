import argparse
import os
import glob
from helpful_functions import safe_open_w, source_names

def readArgs ():
	parser = argparse.ArgumentParser (description="aggregate the statistics from all the random runs")
	parser.add_argument ("--dir-prefix", type=str, required=True, help="prefix of the directory")
	parser.add_argument ("--output-file", type=str, required=True, help="output file")
	parser.add_argument ("--nums", type=str, nargs="+", required=True, help="the numbers for the random runs")
	parser.add_argument ("--thresholds", type=int, nargs="+", required=True, help="the threshold values")
	parser.add_argument ("--lead-types", type=str, nargs="+", required=True, help="the different ways of measuring leads")
	parser.add_argument ("--column-num", type=int, required=False, default=2, help="the column number in the file")
	args = parser.parse_args()
	return args

def extract_source_values (filename, source_names, column_num=2):
	source_values = {source_name: 0.0 for source_name in source_names}
	with open (filename) as fin:
		for i, line in enumerate (fin):
			if i > 0:
				parts = line.strip().split(",")
				src = parts[0]
				if src in source_values:
					source_values[src] = float (parts[column_num])

	return source_values

def main (args):
	with safe_open_w (args.output_file) as fout:
		for thresh in args.thresholds:
			for lead_type in args.lead_types:
				header = ",".join (source_names)
				fout.write (f"{header}\n")
				for num in args.nums:
					filename = os.path.join (args.dir_prefix + f".{num}", f"leader_stats.{thresh}.{lead_type}.csv")
					source_values = extract_source_values (filename, source_names, column_num=args.column_num)
					row = ",".join ([f"{source_values[source_name]:.4f}" for source_name in source_names])
					fout.write (f"{row}\n")	


if __name__ == "__main__":
	main (readArgs ())
