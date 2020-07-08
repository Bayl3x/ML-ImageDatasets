import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import utils


parser = argparse.ArgumentParser(description='Extracts latent semantics of the subject-subject similarity matrix using NMF')
parser.add_argument('model_csv_file', metavar='CSVFILE', help='CSV file with feature vectors matching the images in the given folder')
parser.add_argument('image_folder', metavar='IMAGE_FOLDER', help='Image folder to display from')
parser.add_argument('k', metavar='K', help='The number of latent features to be reported')


args = parser.parse_args()

if not os.path.isfile(args.model_csv_file):
    print("Error: File doesn't exist")
    sys.exit(-1)


feature_model = 'HOG'
#technique = 'SVD' #probably don't use
subjects = utils.get_subjects(args.model_csv_file)
subjectDataMatrixAverages = utils.get_subjects_average_feature_vectors(args.model_csv_file, feature_model, subjects)


subjectDistances = {}
similarityMatrix = []
for subject in subjects:
	dlist = []
	for key in subjectDataMatrixAverages:
		dlist.append(utils.cosine_similarity_vv(subjectDataMatrixAverages[str(subject)], subjectDataMatrixAverages[key]))

	dlist = np.array(dlist, dtype=np.float32)
	subjectDistances[str(subject)] = dlist
	similarityMatrix.append(dlist)

similarityMatrix = np.array(similarityMatrix, dtype=np.float32)

latent_semantics, _ = utils.get_NMF(similarityMatrix, int(args.k))
latent_semantics_with_indices = utils.sort_into_term_weight_pairs(latent_semantics)
latent_semantics_with_subjectIDS = []
for latent_semantic in latent_semantics_with_indices:
	withSubIDS = []
	for tup in latent_semantic:
		withSubIDS.append((subjects[tup[0]], tup[1]))
	latent_semantics_with_subjectIDS.append(withSubIDS)

for ls in latent_semantics_with_subjectIDS:
	print(ls)
	print("")