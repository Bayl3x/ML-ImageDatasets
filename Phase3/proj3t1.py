import argparse
import sys
import os
import proj3t1_utils

parser = argparse.ArgumentParser(description='Given two separate folders with labeled and unlabeled images, and two corresponding csv files paths')
parser.add_argument('labeled_image_folder', metavar='labeled_image_folder', help='The folder with dorsal/palmar labeled images,')
parser.add_argument('csv_labeled', metavar='csv_labeled', help='The path of csv info file of labeled images')
parser.add_argument('unlabeled_image_folder', metavar='unlabeled_image_folder', help='The folder with unlabeled images')
parser.add_argument('-g', '--groundtruth_csv_unlabeled', metavar='groundtruth_csv_unlabeled', help='The path of ground truth csv info file of unlabeled images')
parser.add_argument('-k', '--K', metavar='K_latent_semantics', help='K latent semantics',type=int)
args = parser.parse_args()

if not os.path.isdir(args.labeled_image_folder) :
    print(f"Error: {args.labeled_image_folder} doesn't exist")
    sys.exit(-1)
elif not os.path.isfile(args.csv_labeled):
    print(f"Error: {args.csv_labeled} doesn't exist")
    sys.exit(-1)
elif not os.path.isdir(args.unlabeled_image_folder):
    print(f"Error: {args.unlabeled_image_folder} doesn't exist")
    sys.exit(-1)
if args.groundtruth_csv_unlabeled!=None and not os.path.isfile(args.groundtruth_csv_unlabeled):
    print(f"Error: {args.groundtruth_csv_unlabeled} doesn't exist")
    sys.exit(-1)

proj3t1_utils.executable(args.labeled_image_folder,args.csv_labeled,args.unlabeled_image_folder,args.groundtruth_csv_unlabeled,args.K)
