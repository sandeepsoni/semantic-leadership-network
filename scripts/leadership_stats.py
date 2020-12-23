import pandas as pd
import argparse
import os
import networkx as nx
from collections import Counter
from scipy.stats import binom_test
from helpful_functions import source_names

def readArgs ():
	parser = argparse.ArgumentParser (description="statistics about semantic leaders")
	parser.add_argument ("--src-path", required=True, type=str, help="directory contains the leadership scores")
	parser.add_argument ("--leaders-file", required=True, type=str, help="file contains the leaders and their scores")
	parser.add_argument ("--leader-stats-file", required=True, type=str, help="output file should contain aggregated leadership stats")
	parser.add_argument ("--leader-follower-stats-file", required=True, type=str, help="output file should contain aggregated leadership/followee stats")
	args = parser.parse_args ()
	return args

def create_net (source_names, dyads_counter):
	G = nx.DiGraph ()

	G.add_nodes_from(source_names)
	n_edges = sum([dyads_counter[item] for item in dyads_counter])
	for item in dyads_counter:
		# edge should point from follower to leader
		G.add_edge(item[1], item[0], weight=dyads_counter[item]/n_edges)
	return G

def main (args):
	df = pd.read_csv(os.path.join (args.src_path, args.leaders_file), sep=";")
	rows = df[["s1", "s2"]].values.tolist()
	pairs = list ()
	for row in rows:
		s1, s2 = row[0], row[1]
		pairs.append ((s1, s2))

	leaders_followers = Counter (pairs)

	G = create_net (source_names, leaders_followers)
	pagerank = nx.pagerank_numpy (G, alpha=0.85)
	# include also a personalization factor
	
	leaders = Counter([x1 for x1, x2 in pairs])
	followers = Counter ([x2 for x1, x2 in pairs])
	
	# calculate the leader follower stats
	items = list ()
	total_dyads = sum([item[1] for item in leaders_followers.most_common (None)])
	for item in leaders_followers.most_common (None):
		items.append ([item[0][0], item[0][1], item[1], item[1]/total_dyads])

	leaders_followers_df = pd.DataFrame (items, columns=["Leader", "Follower", "Count", "Probability"])

	# calculate the leader stats
	items = list ()
	epsilon = 1e-10
	for name in sorted (source_names):
		leader_prob = leaders[name]/(leaders[name] + followers[name] + epsilon)
		follower_prob = followers[name]/(leaders[name] + followers[name] + epsilon)
		pval = binom_test([leaders[name], followers[name]], alternative="greater")
		pr = pagerank[name]
		items.append ([name, leaders[name], f"{leader_prob:.4f}", f"{pval:.4f}", followers[name], f"{follower_prob:.4f}", f"{pr:.4f}"])

	leaders_df = pd.DataFrame (items, columns=["Name", "Count(role as leader)", "P(Name=leader)", "Pval", "Count(role as follower)", "P(Name=follower)", "PageRank"])

	## Write to files
	leaders_followers_df.to_csv (os.path.join (args.src_path, args.leader_follower_stats_file), sep=",", index=False, header=True)		
	leaders_df.to_csv (os.path.join (args.src_path, args.leader_stats_file), sep=",", index=False, header=True)

if __name__ == "__main__":
	main (readArgs())
