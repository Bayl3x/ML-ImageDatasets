import argparse
import sys
import os
import matplotlib.pyplot as plt

import utils


parser = argparse.ArgumentParser(description='Identifies and visualizes the 3 most related subjects')
parser.add_argument('subject_id', metavar='SUBJECT', help='Subject ID as in the metadata file')
parser.add_argument('model_csv_file', metavar='CSVFILE', help='CSV file with feature vectors matching the images in the given folder')
parser.add_argument('image_folder', metavar='IMAGE_FOLDER', help='Image folder to display from')

args = parser.parse_args()

if not os.path.isfile(args.model_csv_file):
    print("Error: File doesn't exist")
    sys.exit(-1)


k=100
m = 3
feature_model = 'HOG'
technique = 'SVD'

indices, distances, subjects = utils.get_m_similar_subjects(int(args.subject_id), m, k, args.model_csv_file, technique, feature_model)

#display in text the stuff
for i in indices:
	print("Subject {} had distance {}".format(subjects[i], distances[i]))

#display query subject images
utils.display_subject_images(int(args.subject_id), 0, args.model_csv_file, args.image_folder, feature_model)

#display chosen subject images
for i in range(m):
	utils.display_subject_images(subjects[indices[i]], distances[indices[i]], args.model_csv_file, args.image_folder, feature_model)

