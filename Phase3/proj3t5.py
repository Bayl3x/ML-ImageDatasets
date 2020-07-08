import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LSH import lsh
from LSH import random_vector
import utils
import utils3
from sklearn.decomposition import PCA


# index database

parser = argparse.ArgumentParser(description='Implementation of Task 5')
parser.add_argument('L', metavar='L', help='Number of Layers to use in LSH')
parser.add_argument('K', metavar='K', help='Number of hashes per layer')
parser.add_argument('cm_vectors', metavar='CM_VECTORS', help='Set of CM vectors to index')
parser.add_argument('lbp_vectors', metavar='LBP_VECTORS', help='Set of LBP vectors to index')
# parser.add_argument('query_image', metavar='QUERYIMAGE', help='Query Image')
parser.add_argument('image_folder', metavar='IMAGEFOLDER', help='Folder that stores all images in the vector file')
parser.add_argument('t', metavar='T', help='number of similar images to get')
parser.add_argument('r', metavar='R', help='Value used in hashing, 100 is a good choice')
parser.add_argument('num_buckets', metavar='NUM_BUCKETS', help='Number of buckets used in each hash. 100 seems good')
parser.add_argument('distance_measure', metavar='DISTANCE_MEASURE', help='Manhatten or Euclidean distance')
parser.add_argument('-s', '--showImages', help='Show most similar images', action='store_true')


args = parser.parse_args()

n_components_CM = 200
n_components_LBP = 112


print('Getting Image Data...')
images = utils.get_image_names(args.cm_vectors, 'CM')
cm_data = utils.get_data_matrix(args.cm_vectors, 'CM')
lbp_data = utils.get_data_matrix(args.lbp_vectors, 'LBP')
cm_pca = PCA(n_components=n_components_CM)
lbp_pca = PCA(n_components=n_components_LBP)
data = cm_pca.fit_transform(cm_data)
data2 = lbp_pca.fit_transform(lbp_data)
data = np.concatenate((data, data2), axis=1)
del data2






print('Getting Range for RandomVectorGenerator class...')
r = utils3.get_range(data)
# # get val_range from the feature
# if args.feature_model == 'CM':
#     r = utils3.get_CM_range()
# else:
#     r = utils3.get_range(data[0])
#     r = random_vector.ValueRanges(1, [r[1]], [r[0]])

print('Creating LSH object...')
LSH = lsh.LocalitySensitiveHashing(l=int(args.L), k=int(args.K), r=int(args.r), length=len(data[0]), val_range=r, num_buckets_per_hash=int(args.num_buckets))

print('Building LSH Index...')
index = utils3.create_LSH_index(LSH, data, images)

while True:
    query_image = input(">>")
    query_image=os.path.basename(query_image)

    if query_image == 'exit':
        break

    query_image_base=os.path.basename(query_image)

    if not os.path.exists(os.path.join(args.image_folder, query_image_base)):
        print('File does not exist')
        continue

    #image = utils.load_image(os.path.join(args.image_folder, query_image_base))

    if query_image_base in images:
        q_index = np.where(images == query_image_base)[0][0]
        query_image_fv = data[q_index]
    else:
        query_img = utils.load_image(query_image)
        query_image_fv2_cm = utils3.get_CM(query_img).reshape(1, -1)
        query_image_fv2_lbp = utils3.get_LBP(query_img).reshape(1, -1)
        query_image_fv2_cm = cm_pca.transform(query_image_fv2_cm)
        query_image_fv2_lbp = lbp_pca.transform(query_image_fv2_lbp)
        query_image_fv = np.concatenate((query_image_fv2_cm, query_image_fv2_lbp), axis=1)[0]





    # if args.feature_model == 'CM':
    #     query_image_fv = np.array(utils3.get_CM(image))
    # elif args.feature_model == 'LBP':
    #     query_image_fv = np.array(utils3.get_LBP(image))
    # elif args.feature_model == 'CMLBP':
    #     query_image_fv = np.array(utils3.get_CMLBP(image))


    t_images = utils3.get_similar_from_LSH(LSH, query_image_fv, index, int(args.t), images, data, args.distance_measure)


    #display query image
    if args.showImages:
        plt.imshow(utils.load_image_rgb(os.path.join(args.image_folder, query_image)))
        plt.title('Query image')
        plt.show()


    print(f"len: {len(t_images)}")
    for image in t_images:
        print("Image: {}, Score: {}".format(image[0], image[1]))
        if args.showImages:
            plt.imshow(utils.load_image_rgb(os.path.join(args.image_folder, image[0])))
            plt.title(image[0])
            plt.show()

    # print(f"Buckets: {LSH.buckets_used}")
    # print(f"Total buckets posible: {len(LSH.buckets_used)}")
