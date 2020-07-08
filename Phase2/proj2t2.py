import argparse
import sys
import os
import matplotlib.pyplot as plt

import utils



parser = argparse.ArgumentParser(description='Finds m most similar images using latent semantics')
parser.add_argument('model_csv_file', metavar='CSVFILE', help='CSV file that contains the feature vectors for the desired feature model')
parser.add_argument('feature_model', metavar='MODEL', help='Feature model that is used in the given CSV (CM, LBP, HOG, SIFT)')
parser.add_argument('image_name', metavar='IMAGE_NAME', help='Query image name, don\'t include the path (assumed to also be in the given CSV)')
parser.add_argument('image_folder', metavar='IMAGE_FOLDER', help='Directory for which the images are stored')
parser.add_argument('k', metavar='K', help='The number of latent features to be reported')
parser.add_argument('technique', metavar='TECHNIQUE', help='technique to perform feature extraction (PCA, SVD, NMF, LDA)')
parser.add_argument('m', metavar='M', help='The number of similar images desired')

args = parser.parse_args()


if not os.path.isfile(args.model_csv_file):
    print("Error: File doesn't exist")
    sys.exit(-1)

#get query image feature vector
query_image_feature_vector = utils.get_individual_feature_vector(args.model_csv_file, args.image_name, args.feature_model)

#get the dataMatrix
dataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model)

#normalize CM if technique is LDA and NMF
if args.feature_model == 'CM':
    if args.technique == 'LDA' or args.technique == 'NMF':
        dataMatrix = utils.normalize_cm(dataMatrix)

#if using SIFT, need to reorder the keypoints so that they
#match as much as possible. Here we can use the query image
#as the image we want to match
#if args.feature_model == 'SIFT':
	#dataMatrix = utils.reorder_keypoints(dataMatrix, dataMatrix[0])
	#reorder_keypoints turned out to be a failed endeavor

#weight the dataMatrix
weightMatrix, _ = utils.extract_k_latent_semantics(dataMatrix, int(args.k), args.technique, args.feature_model)
dataMatrix = utils.apply_weights(dataMatrix, weightMatrix, args.feature_model)

#weight the feature vector of the query image
query_image_feature_vector = utils.apply_weights(query_image_feature_vector, weightMatrix, args.feature_model, matrix_wise=False)

#get all the image names for displaying purposes
image_names_in_data_matrix = utils.get_image_names(args.model_csv_file, args.feature_model)

#actually find the most similar images
indices, distances = utils.get_closest_images(dataMatrix, query_image_feature_vector, int(args.m), args.feature_model, int(args.k))

plt.imshow(utils.load_image_rgb(os.path.join(args.image_folder, args.image_name)))
plt.show()
for i in indices:
	print('{}\t{}'.format(image_names_in_data_matrix[i], distances[i]))
for i in indices:
	plt.imshow(utils.load_image_rgb(os.path.join(args.image_folder, image_names_in_data_matrix[i])))
	plt.show()
