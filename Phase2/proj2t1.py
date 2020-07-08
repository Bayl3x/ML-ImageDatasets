import argparse
import sys
import os
import numpy as np

import utils



parser = argparse.ArgumentParser(description='Extracts k latent semantics')
parser.add_argument('model_csv_file', metavar='CSVFILE', help='CSV file that contains the feature vectors for the desired feature model')
parser.add_argument('feature_model', metavar='MODEL', help='Feature model that is used in the given CSV (CM, LBP, HOG, SIFT)')
parser.add_argument('k', metavar='K', help='The number of latent features to be reported')
parser.add_argument('technique', metavar='TECHNIQUE', help='technique to perform feature extraction (PCA, SVD, NMF, LDA)')
parser.add_argument('image_folder', metavar='IMAGE_FOLDER', help='directory that the images are stored in, must be exactly those stored in the CSV')
parser.add_argument('extra_credit', metavar='EXTRA_CREDIT', help='Inlude the extra credit plots. Gets messy if k or n is large (Not working for SIFT)')

args = parser.parse_args()


if not os.path.isfile(args.model_csv_file):
    print("Error: File doesn't exist")
    sys.exit(-1)

#get the dataMatrix from the CSV
dataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model)

#normalize CM if technique is LDA and NMF
if args.feature_model == 'CM':
    if args.technique == 'LDA' or args.technique == 'NMF':
        dataMatrix = utils.normalize_cm(dataMatrix)

#get the k latent semantics
fls, dls = utils.extract_k_latent_semantics(dataMatrix, int(args.k), args.technique, args.feature_model)

#get image names
inames = utils.get_image_names(args.model_csv_file, args.feature_model)

#display the two semantic spaces
utils.display_term_weight_pairs(fls, dls, inames, args.feature_model)

#extra credit
if args.extra_credit == 'true':
	utils.display_dls(dataMatrix, dls, inames, args.image_folder) #note: do not do with datasets of large size because every image is shown for each latent semantic
	utils.display_fls(dataMatrix, fls, inames, args.image_folder)
