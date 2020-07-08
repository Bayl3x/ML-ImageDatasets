import argparse
import sys
import os
import matplotlib.pyplot as plt

import utils



parser = argparse.ArgumentParser(description='Classifies a given image using latent semantics from a subset of the data')
parser.add_argument('model_csv_file', metavar='CSVFILE', help='CSV file that contains the feature vectors for the desired feature model')
parser.add_argument('feature_label', metavar='LABEL', help='feature_label to perform feature extraction (left, right, palmar, dorsal, noAccessories, accessories, male, female)')
parser.add_argument('feature_model', metavar='MODEL', help='Feature model that is used in the given CSV (CM, LBP, HOG, SIFT)')
parser.add_argument('image_name', metavar='IMAGE_NAME', help='Query image name, don\'t include the path (assumed to also be in the given CSV)')
parser.add_argument('image_folder', metavar='IMAGE_FOLDER', help='Directory for which the images are stored')
parser.add_argument('k', metavar='K', help='The number of latent features to be reported')
parser.add_argument('technique', metavar='TECHNIQUE', help='technique to perform feature extraction (PCA, SVD, NMF, LDA)')

args = parser.parse_args()

if not os.path.isfile(args.model_csv_file):
    print("Error: File doesn't exist")
    sys.exit(-1)

classificationResults = []
query_image_feature_vector = utils.get_individual_feature_vector(args.model_csv_file, args.image_name, args.feature_model)

#parameters for testing
qRatio = 20 	#value of K for KNN (called q cuz k means something else)

weightMethod = 1	#distance or frequency
method = 2	#KNN or Center based clustering

#KNN
if(method == 2):	
	#get the latent semantics associated with the given label
	dataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label=args.feature_label)
	weightMatrix, _ = utils.extract_k_latent_semantics(dataMatrix, int(args.k), args.technique, args.feature_model)
	#apply latent semantics to entire dataset
	dataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model)
	dataMatrix = utils.apply_weights(dataMatrix, weightMatrix, args.feature_model)

	#get the query image feature vector in the latent space
	query_image_feature_vector = utils.apply_weights(query_image_feature_vector, weightMatrix, args.feature_model, matrix_wise=False)

	q = int(len(dataMatrix) / qRatio)

	#find the q closest neighbors to the query image
	ind, ds = utils.get_closest_images(dataMatrix, query_image_feature_vector, q, args.feature_model, int(args.k))
	#only keep dists that are for the returned images
	dists = []
	for i in ind:
		dists.append(ds[i])
	del ds

	#get metadata for the images
	imageData = utils.get_image_names_and_labels(args.model_csv_file, args.feature_model)

	if weightMethod == 1:
		#weighting with inverse of distances
		votes = utils.knn_distance_weights(ind, dists, imageData)
	else:
		#weighting with inverse of frequency
		votes = utils.knn_frequency_weights(ind, dists, imageData)

	classifications, percentages = utils.classify_votes(votes)
	for i in range(len(classifications)):
		print("{}\t {}%".format(classifications[i], percentages[i]))

#averages
else:
	#get the latent semantics associated with the given label
	dataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label=args.feature_label)
	
    #normalize CM if technique is LDA and NMF
    if args.feature_model == 'CM':
        if args.technique == 'LDA' or args.technique == 'NMF':
            dataMatrix = utils.normalize_cm(dataMatrix)
    
    weightMatrix, _ = utils.extract_k_latent_semantics(dataMatrix, int(args.k), args.technique, args.feature_model)
	del dataMatrix

	#get the query image feature vector in the latent space
	query_image_feature_vector = utils.apply_weights(query_image_feature_vector, weightMatrix, args.feature_model, matrix_wise=False)

	#start comparing between the 4 binary classes

	#get left average distance
	leftDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='left')
	#put it into latent space
	leftDataMatrix = utils.apply_weights(leftDataMatrix, weightMatrix, args.feature_model)
	leftAvgD = utils.get_average_distance(leftDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del leftDataMatrix

	#get right average distance
	rightDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='right')
	#put it into latent space
	rightDataMatrix = utils.apply_weights(rightDataMatrix, weightMatrix, args.feature_model)
	rightAvgD = utils.get_average_distance(rightDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del rightDataMatrix

	if leftAvgD < rightAvgD:
		classificationResults.append('left')
	else:
		classificationResults.append('right')


	#get dorsal average distance
	dorsalDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='dorsal')
	#put it into latent space
	dorsalDataMatrix = utils.apply_weights(dorsalDataMatrix, weightMatrix, args.feature_model)
	dorsalAvgD = utils.get_average_distance(dorsalDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del dorsalDataMatrix

	#get palmar average distance
	palmarDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='palmar')
	#put it into latent space
	palmarDataMatrix = utils.apply_weights(palmarDataMatrix, weightMatrix, args.feature_model)
	palmarAvgD = utils.get_average_distance(palmarDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del palmarDataMatrix

	if dorsalAvgD < palmarAvgD:
		classificationResults.append('dorsal')
	else:
		classificationResults.append('palmar')


	#get left average distance
	accessoriesDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='accessories')
	#put it into latent space
	accessoriesDataMatrix = utils.apply_weights(accessoriesDataMatrix, weightMatrix, args.feature_model)
	accessoriesAvgD = utils.get_average_distance(accessoriesDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del accessoriesDataMatrix

	#get right average distance
	noAccessoriesDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='noAccessories')
	#put it into latent space
	noAccessoriesDataMatrix = utils.apply_weights(noAccessoriesDataMatrix, weightMatrix, args.feature_model)
	noAccessoriesAvgD = utils.get_average_distance(noAccessoriesDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del noAccessoriesDataMatrix

	if accessoriesAvgD < noAccessoriesAvgD:
		classificationResults.append('accessories')
	else:
		classificationResults.append('noAccessories')


	#get male average distance
	maleDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='male')
	#put it into latent space
	maleDataMatrix = utils.apply_weights(maleDataMatrix, weightMatrix, args.feature_model)
	maleAvgD = utils.get_average_distance(maleDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del maleDataMatrix

	#get right average distance
	femaleDataMatrix = utils.get_data_matrix(args.model_csv_file, args.feature_model, label='female')
	#put it into latent space
	femaleDataMatrix = utils.apply_weights(femaleDataMatrix, weightMatrix, args.feature_model)
	femaleAvgD = utils.get_average_distance(femaleDataMatrix, int(args.k), args.feature_model, query_image_feature_vector)
	del femaleDataMatrix

	if maleAvgD < femaleAvgD:
		classificationResults.append('male')
	else:
		classificationResults.append('female')
	print(classificationResults)
