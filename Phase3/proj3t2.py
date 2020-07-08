import argparse
import sys
import os

import proj3t2_util



parser = argparse.ArgumentParser(description='Identify the image by cluster descriptor.')
parser.add_argument('labelled_image_folder', metavar='labelledFOLDER', help='Folder with labelled images')
parser.add_argument('unlabelled_image_folder', metavar='unlabelledFOLDER', help='Folder with unlabelled images')

# parser.add_argument('feature_model', metavar='MODEL', help='Feature model to use')
parser.add_argument('info_file', metavar='INFOFILE', help='metadata file for Hands database')
# parser.add_argument('output_filename', metavar='OUTPUT', help='name of output file')

# parser.add_argument('traning_feature_csv', metavar='TraningFeatureCSV', help='Training file with image feature vector')
# parser.add_argument('testing_feature_csv', metavar='TestingFeatureCSV', help='Testing file with image feature vector')
parser.add_argument('number_of_cluster_for_palmar', metavar='PalmarNum', help='Specify how many clusters for the palmar images')
parser.add_argument('number_of_cluster_for_dorsal', metavar='DorsalNum', help='Specify how many clusters for the dorsal images')

args = parser.parse_args()

proj3t2_util.main(args.labelled_image_folder,args.unlabelled_image_folder, args.info_file, int(args.number_of_cluster_for_palmar), int(args.number_of_cluster_for_dorsal))
