 
import argparse
import sys
import os

import utils



parser = argparse.ArgumentParser(description='Extracts features from all images in given folder')
parser.add_argument('image_folder', metavar='FOLDER', help='Directory with images from which features should be extracted')
parser.add_argument('feature_model', metavar='MODEL', help='Feature model to use')
parser.add_argument('info_file', metavar='INFOFILE', help='metadata file for Hands database')
parser.add_argument('output_filename', metavar='OUTPUT', help='name of output file')

args = parser.parse_args()


if not os.path.isdir(args.image_folder):
    print("Error: Directory doesn't exist")
    sys.exit(-1)

utils.process_images(args.image_folder, args.output_filename, args.feature_model, args.info_file)