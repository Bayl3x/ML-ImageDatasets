import argparse
import sys
import os
import pandas as pd
import numpy as np
from dtree import dtree
from SVM import SVM
from ppr import PPR

import utils3
from sklearn.decomposition import PCA, TruncatedSVD

parser = argparse.ArgumentParser(description='Creates a classifying model from one folder of images, then makes predictions on another folder of images.')
parser.add_argument('model', metavar='MODEL', help='model to be used (SVM, DTREE, PPR)')
parser.add_argument('input_folder', metavar='INPUT_FOLDER', help='Folder of input images used to train the model')
parser.add_argument('test_folder', metavar='TEST_FOLDER', help='Folder of images to have labels predicted')
parser.add_argument('train_image_metadata', metavar='METADATA_INPUT', help='File of metadata to extract labels for training set')
parser.add_argument('test_image_metadata', metavar='METADATA_TEST', help='File of metadata to extract labels for testing set ("N" for test data without labels)')

args = parser.parse_args()

if args.model == 'DTREE':
    kratio = 1 / 2
    feature_model = 'CM'
    dimTech = 'PCA'
elif args.model == 'SVM':
    kratio = 1 / 3
    feature_model = 'CM'
    dimTech = 'PCA'
else:
    kratio = 1 / 2
    feature_model = 'LBP'
    dimTech = 'PCA'

#need feature matrices of the input_folder. Don't think this is storable...
train_feature_vectors, image_names = utils3.get_feature_vectors(args.input_folder, feature_model)


k = int(len(train_feature_vectors) * kratio)


if k > len(train_feature_vectors):
	k = len(train_feature_vectors) - 1

#apply dimensionality reduction on the input
if dimTech == 'SVD':
	pca = TruncatedSVD(n_components=k)
else:
	pca = PCA(n_components=k)
pca.fit(train_feature_vectors)
train_feature_vectors = pca.transform(train_feature_vectors)

#get train image labels
train_labels = utils3.get_labels(image_names, args.train_image_metadata)

train_df = pd.DataFrame(train_feature_vectors)
train_df['image_name'] = image_names
train_df['label'] = train_labels


#we now have:
#1) data frame that has:
#	a) image names stored in second to last column called 'image_name'
#	b) latent feature matrix stored in columns 0-k
#	c) target values stored in the last column called 'label'
#2) PCA model that we'll use to transform the test features as well

#can begin to train the model
if args.model == 'DTREE':
	model = dtree(max_depth=None)
elif args.model == 'SVM':
	model = SVM(kernel='poly', kernel_param = 3, soft_margin = 0.5)
else:
	model = PPR(k=5, image_names=image_names)

x_train = np.array(train_df.values[:, 0:k])
y_train = train_df['label'].values
model.fit(x_train, y_train)



#now we load the test data to predict on
test_feature_vectors, image_names = utils3.get_feature_vectors(args.test_folder, feature_model)
test_feature_vectors = pca.transform(test_feature_vectors)

if args.test_image_metadata == 'N':
    test_labels = None
else:
    test_labels = utils3.get_labels(image_names, args.test_image_metadata)

test_df = pd.DataFrame(test_feature_vectors)
test_df['image_name'] = image_names
test_df['label'] = test_labels
if args.model == 'PPR':
    model.test_names = image_names
x_test = np.array(test_df.values[:, 0:k])
y_test = test_df['label'].values
y_pred = model.predict(x_test)

print("Image\t\t\t\tClassified as\t\tActual")
print("-------------------------------------------------------------------------------")
for i in range(len(image_names)):
	print("{}\t\t{}\t\t\t{}".format(image_names[i], y_pred[i], y_test[i]))


#calculate accuracy
total_correct = 0
for i in range(len(y_pred)):
	if(y_test[i] == y_pred[i]):
		total_correct += 1
if test_labels != None:
    print("{} out of {} correctly classified. Accuracy: {}".format(total_correct, len(y_pred), str(total_correct/len(y_pred))))
