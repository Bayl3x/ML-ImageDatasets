import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import utils


parser = argparse.ArgumentParser(description='Extracts latent semantics of the binary metadata matrix using NMF')
parser.add_argument('metadata_csv_file', metavar='METADATA', help='CSV file that contains the metadata, \'/Path/HandInfo.csv\'')
parser.add_argument('k', metavar='K', help='The number of latent features to be reported')


args = parser.parse_args()

if not os.path.isfile(args.metadata_csv_file):
    print("Error: File doesn't exist")
    sys.exit(-1)


bmm, inames, mdfeatures = utils.get_binary_metadata_matrix(args.metadata_csv_file)


latent_semantics, _ = utils.get_NMF(bmm.T, int(args.k))
latent_semantics_with_indices = utils.sort_into_term_weight_pairs(latent_semantics)
latent_semantics_with_inames = []
for latent_semantic in latent_semantics_with_indices:
	withinames = []
	for tup in latent_semantic:
		withinames.append((inames[tup[0]], tup[1]))
	latent_semantics_with_inames.append(withinames)

for ls in latent_semantics_with_inames:
	print(ls)
	print("")


latent_semantics, _ = utils.get_NMF(bmm, int(args.k))
latent_semantics_with_indices = utils.sort_into_term_weight_pairs(latent_semantics)
latent_semantics_with_mdfeatures = []
for latent_semantic in latent_semantics_with_indices:
	withmdfeatures = []
	for tup in latent_semantic:
		withmdfeatures.append((mdfeatures[tup[0]], tup[1]))
	latent_semantics_with_mdfeatures.append(withmdfeatures)

for ls in latent_semantics_with_mdfeatures:
	print(ls)
	print("")

