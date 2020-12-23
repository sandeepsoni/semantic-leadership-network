import argparse
import pandas as pd
import os
import numpy as np
import logging
logging.basicConfig (format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

def readArgs ():
	parser = argparse.ArgumentParser (description="leadership score calculation")
	parser.add_argument ("--src-path", type=str, required=True, help="directory that contains the source embeddings")
	parser.add_argument ("--leaders-file", type=str, required=True, help="file that contains the leadership scores")
	parser.add_argument ("--thresh", type=int, required=False, default=50, help="threshold in terms of percentile (default: 50)")
	parser.add_argument ("--thresholds-file", type=str, required=True, help="file that contains the leadership scores but filtered to satisfy threshold")
	parser.add_argument ("--lead-type", type=str, required=False, default="l1", choices={"l1", "l2", "l3", "l4", "l5"}, help="file that contains the leadership scores but filtered to satisfy threshold")
	args = parser.parse_args ()
	return args

def main (args):
	# read the leadership scores file
	lead_short_codes = {"l1": "Lead1", "l2": "Lead2", "l3": "Lead3", "l4": "Lead4", "l5": "Lead5"}
	lead_key = lead_short_codes[args.lead_type]

	src_file = os.path.join (args.src_path, args.leaders_file)
	df = pd.read_csv (src_file, sep=";")
	header = df.columns.values.tolist()
	keep_cols = ["word", "rank", "Period1", "Period2", "Neighbors1", "Neighbors2", "Freq1", "Freq2", "NeighborsFreq1", "NeighborsFreq2"]
	cols = [col for col in header if col in keep_cols or col.startswith (lead_key)]
	df = df[cols]
	header = df.columns.values.tolist ()
	remapped = {col: col if not col.startswith (lead_key) or not "_" in col else col.split("_")[1] for col in header}
	df = df.rename (columns=remapped)	
	logging.info (f"Read leadership values from {src_file}: {len(df)} rows")
	lead = df[lead_key].values
	val = np.percentile (lead, args.thresh)
	logging.info (f"Calculated {args.thresh} percentile as {val:.6f}")
	df = df[df[lead_key] >= val]
	logging.info (f"Retained rows after threshold filter {len(df)}")
	tgt_file = os.path.join (args.src_path, args.thresholds_file)
	df.to_csv (tgt_file, sep=";", header=True, index=False)
	logging.info (f"Filtered dyads written to {tgt_file}")

if __name__ == "__main__":
	main (readArgs ())
