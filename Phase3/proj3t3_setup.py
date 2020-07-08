import argparse
import sys
import os
from sklearn.decomposition import PCA

import utils, utils3



parser = argparse.ArgumentParser(description='Creates image-image similarity dense matrix')
parser.add_argument('input_folder', metavar='INPUT_FOLDER', help='Folder of images from which feature vectors should be extracted')
parser.add_argument('output_file', metavar='OUTPUT', help='Output location of the image image similarity matrix to be stored')

args = parser.parse_args()

feature_vectors, image_names = utils3.get_feature_vectors(args.input_folder, 'LBP')

kratio = 1 / 2

k = int(len(feature_vectors) * kratio)


if k > len(feature_vectors):
	k = len(feature_vectors) - 1

pca = PCA(n_components=k)
pca.fit(feature_vectors)
feature_vectors = pca.transform(feature_vectors)


utils3.compute_and_store_ii_similarity(feature_vectors, image_names, args.output_file)
