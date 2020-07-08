import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD

from LSH import lsh
import utils
import utils3


parser = argparse.ArgumentParser(description='Implementation of Task 6')
parser.add_argument('rfs_type', metavar='RFSTYPE', help='Type of RFS system to use (SVM, DTREE, PPR, PROB)')
parser.add_argument('vectors', metavar='VECTORS', help='Set of CM vectors to index')
parser.add_argument('vectors2', metavar='VECTORS2', help='Set of LBP vectors to index')
parser.add_argument('query_image', metavar='QUERYIMAGE', help='Query Image')
parser.add_argument('image_folder', metavar='IMAGEFOLDER', help='Folder that stores all images in the vector file')
parser.add_argument('t', metavar='T', help='number of similar images to get')
parser.add_argument('display', metavar='DISPLAY', help='true of false to display similar images when choosing relevant or irrelevant')
parser.add_argument('l', metavar='L', help='The number of hash families')
parser.add_argument('k', metavar='K', help='the number of hash functions per family')


args = parser.parse_args()

#tunable params
L = int(args.l)
K = int(args.k)
r_val = 1000
num_buckets = 5
n_components_CM = 200
n_components_LBP = 112

#initial things
query_image=os.path.basename(args.query_image)
entire_image_names = utils.get_image_names(args.vectors, 'CM')
entire_data = utils.get_data_matrix(args.vectors, 'CM')
entire_data2 = utils.get_data_matrix(args.vectors2, 'LBP')
if query_image in entire_image_names:
    q_index = np.where(entire_image_names == query_image)[0][0]
else:
    #get query image feature vectors
    query_img = utils.load_image(args.query_image)
    query_image_fv2_cm = utils3.get_CM(query_img).reshape(1, -1)
    query_image_fv2_lbp = utils3.get_LBP(query_img).reshape(1, -1)

pca = PCA(n_components=n_components_CM)
entire_data = pca.fit_transform(entire_data)

pca2 = PCA(n_components=n_components_LBP)
entire_data2 = pca2.fit_transform(entire_data2)
entire_data = np.concatenate((entire_data, entire_data2), axis=1)
del entire_data2

if not query_image in entire_image_names:
    query_image_fv2_cm = pca.transform(query_image_fv2_cm)
    query_image_fv2_lbp = pca2.transform(query_image_fv2_lbp)
    query_image_fv = np.concatenate((query_image_fv2_cm, query_image_fv2_lbp), axis=1)
else:
    query_image_fv = entire_data[q_index]



# get val_range from the feature
r = utils3.get_range(entire_data)
LSH = lsh.LocalitySensitiveHashing(l=L, k=K, r=r_val, length=len(entire_data[0]), val_range=r, num_buckets_per_hash=num_buckets)
index = utils3.create_LSH_index(LSH, entire_data, entire_image_names)

    

def get_t_most_similar_from_LSH(pquery_image, pdata, pimage_names):
    t_images = utils3.get_similar_from_LSH(LSH, pquery_image, index, int(args.t), pimage_names, pdata, 'euclidean')
    return t_images



#initial run
data = entire_data
image_names = entire_image_names
images = get_t_most_similar_from_LSH(query_image_fv, data, image_names)
relevant_images = []
irrelevant_images = []

#the query image is given as relevant
relevant_images.append([query_image, query_image_fv])

keepGoing = True
keepLabelling = True
keepDisplaying = 'y'
while keepGoing:
    print("Please label the following images as relevant (r) or irrelevant(i): (type q if you'd like to be done labelling)")

    for image in images:
        if args.display.upper() == 'TRUE' and keepDisplaying == 'y':
            plt.imshow(utils.load_image_rgb(os.path.join(args.image_folder, image[0])))
            plt.title(image[0])
            plt.show()

        if keepLabelling:
            response = input("{} score: {}labeled as: ".format(image[0], image[1]))
            if response == 'r':
                if image[0] not in utils3.column(relevant_images, 0):
                    i = np.where(image_names == image[0])
                    relevant_images.append([image_names[i][0], data[i][0]])

                    #remove from irrelevant if user changed their mind
                    if image[0] in utils3.column(irrelevant_images, 0):
                        index_del = np.where(np.array(utils3.column(irrelevant_images, 0)) == image[0])
                        del irrelevant_images[index_del[0][0]]
            elif response == 'i':
                if image[0] not in utils3.column(irrelevant_images, 0):
                    i = np.where(image_names == image[0])
                    irrelevant_images.append([image_names[i][0], data[i][0]])

                    #remove from relevant if user changed their mind
                    if image[0] in utils3.column(relevant_images, 0):
                        index_del = np.where(np.array(utils3.column(relevant_images, 0)) == image[0])
                        del relevant_images[index_del[0][0]]
            elif response == 'q':
                keepLabelling = False
                if args.display.upper() == 'TRUE':
                    keepDisplaying = input("Would you like to display the rest of the images? (y/n)")
        else:
            print("{} score: {}".format(image[0], image[1]))

    if input("Would you like to continue the RFS? (y/n)") == 'n':
        keepGoing = False
        break
    else:
        keepDisplaying = 'y'
        keepLabelling = True



    #modify data, or change ordering of returned results
    #note: if you are changing the data, you must change the image_names to be ordered the same way
    if args.rfs_type == 'DTREE' or args.rfs_type == 'SVM' or args.rfs_type == 'PPR':
        #data, image_names = utils3.prune_irrelevant(args.rfs_type, entire_data, entire_image_names, relevant_images, irrelevant_images)
        query_image_fv = utils3.modify_query(args.rfs_type, entire_data, entire_image_names, relevant_images, irrelevant_images, query_image_fv)
        #re-get the images based on a new data, or a new query image
        images = get_t_most_similar_from_LSH(query_image_fv, data, image_names)
    else:
        images = utils3.reorder_t_results_probabilisticly(images, data, image_names, relevant_images, irrelevant_images)





if input("Would you like to redisplay the final images? (y/n)") == 'y':
    print("The selected {} most similar images after RFS:".format(args.t))
    for image in images:
        print("Image: {}, Score: {}".format(image[0], image[1]))
        if args.display.upper() == 'TRUE':
            plt.imshow(utils.load_image_rgb(os.path.join(args.image_folder, image[0])))
            plt.title(image[0])
            plt.show()
