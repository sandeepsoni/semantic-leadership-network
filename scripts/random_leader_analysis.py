import argparse
import os
import pandas as pd
from helpful_functions import source_names

def readArgs ():
	parser = argparse.ArgumentParser (description="randomized trials data")
	parser.add_argument ("--dir-path", type=str, required=True, help="directory path")
	parser.add_argument ("--prefix", type=str, required=True, help="prefix of the directories that contain the relevant files")
	parser.add_argument ("--runs", type=int, required=True, nargs="+", help="integer for num")
	parser.add_argument ("--leader-stats-file", type=str, required=True, help="leader stats filename")
	parser.add_argument ("--field-name", type=str, required=False, default="P(Leader)", choices={"P(Leader)", "PageRank"}, help="field name to extract")
	parser.add_argument ("--out-file", type=str, required=True, help="csv file containing the results from all the runs for the given field name")
	args = parser.parse_args ()
	return args

def main (args):
	rows = list ()
	rows.append (sorted (source_names))
	for run in args.runs:
		filename = os.path.join (args.dir_path, f"{args.prefix}.{run}", args.leader_stats_file)
		df = pd.read_csv (filename)
		items = [df[df["Leader"] == name][args.field_name].iloc[0] for name in sorted (source_names)]
		rows.append (items)
	out_df = pd.DataFrame (rows)
	out_df.to_csv (args.out_file, header=False, index=False)

if __name__ == "__main__":
	main (readArgs ())
