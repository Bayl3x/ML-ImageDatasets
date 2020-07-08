import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
import math
import utils
import glob
import cv2
from scipy.stats import skew, multivariate_normal
from mahotas.features.lbp import lbp
from LSH import random_vector
from LSH import lsh
from collections import Counter
import sys
import matplotlib.pyplot as plt
from dtree import dtree
from SVM import SVM
from ppr import PPR

import random

def column(matrix, i):
    return [row[i] for row in matrix]

#just to split utils files cuz other one long

def get_latent_data(csv_file, k=100):
    df = pd.read_csv(csv_file, header=None) 

    if k > len(df):
        k = len(df) - 1
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(df.values[:, 6:])

#stores the image-image similarity matrix
#so in future we can modify it to only have K values in task 3 and
#set the rest to 0, thus simulating a graph of sorts
def compute_and_store_ii_similarity(dm, inames, output_file):
    dmdict = {}
    length = len(inames)
    #calculate cosine similarity for all of images for all images
    for i in range(length):
        dmdict[inames[i]] = utils.cosine_similarity_mv(dm, dm[i])
        print("{} remaining".format(length - i))
    #create a data frame with correct column and row names corresponding to the images
    df = pd.DataFrame.from_dict(dmdict, orient='index')
    df.columns = inames

    #write to csv
    df.to_csv(output_file)

#get top k similarty graph for each images.
#uses the ii_similarity matrix, sort by distance, get top K, and prune the rest
# also getting and returning the adjacency matrix here since the same values 
#  for the graph can be used to generate the matrix
def get_k_similartiy_graph(similarityMatrix, k):
    
    graph = {}

    #offset by 1 to get itself + k images
    k = k + 1
    node = [None] * k
    adjacencyMatrix = np.zeros([len(similarityMatrix), len(similarityMatrix)])
    for i in (similarityMatrix):
        #image index sorted by distance desc
        image_index = np.argsort(similarityMatrix[i].values)[::-1][0:k]
        #image distance sorted desc
        edge = np.sort(similarityMatrix[i].values)[::-1][0:k]
        #get image name using image index
        #+1 because index 0 gives "unamed:0"
        for j in range(k):
            node[j] = similarityMatrix.columns[image_index[j]]
            if j > 0:
                adjacencyMatrix[image_index[0]][image_index[j]] = edge[j]

            
        #node[0] is itself (distance = 1, root)
        #node[1:k] are the top k images (leaves with corresponding edge weights)
        graph[node[0]] = (node[1:k], edge[1:k])
    

    """
    for i in range(len(adjacencyMatrix)):
        find = np.where(adjacencyMatrix[i] == 1)
        print(i, find)
    """
    return graph, adjacencyMatrix

# Perform PPR on the top k similarity graph using 1 or 3 image IDs
def ppr(graph, image_names, adjacencyMatrix, image1, image2=None, image3=None):
    # Pseudocode step 1: Restart vector should be initialized to all 0s except for 1 in position of query image
    restart_vector = np.zeros(len(image_names))
    
    # Multiple image IDs have been passed in for task 3
    if image2 is not None:
        image_index1 = np.where(image_names == image1)
        image_index2 = np.where(image_names == image2)
        image_index3 = np.where(image_names == image3)
    
        restart_vector[image_index1] = (1/3)
        restart_vector[image_index2] = (1/3)
        restart_vector[image_index3] = (1/3)

    # One image ID has been passed in for task 4
    else:
        image_index = np.where(image_names == image1)
        restart_vector[image_index] = 1

    
    # Pseudocode step 2: Normalize columns of Adjacency matrix so that they sum to 1
    #  Columns that add up to 0 originally are kept as 0; that's the approach I saw when I briefly looked into this issue
    col_sums = adjacencyMatrix.sum(axis=0,keepdims=1)
    col_sums[col_sums==0] = 1
    normal_adj = adjacencyMatrix/col_sums
    
    # Pseudocode step 3: Initialize state probability vector to initial value of restart vector
    state_prob_vector = restart_vector
    
    # Pseudocode step 4: While state prob vector has not converged, perform calculation below
    # Paper also says that a maximum number of iterations can be specified; will do this for now
    #  because I'm uncertain of how to determine when the vector has converged at this point
    #  Also, c value (probability of restart) is something we may want to try different values with; paper recommends between 0.8 and 0.9
    maxIter = 50
    iterCount = 0
    c = 0.85
    while iterCount < maxIter:
        state_prob_vector = (1-c)*normal_adj.dot(state_prob_vector) + (c*restart_vector)
        iterCount = iterCount + 1
    return state_prob_vector

def get_feature_vectors(folder, feature_model='LBP'):
    images = glob.glob1(folder,"*.jpg")
    feature_vectors = []
    if feature_model == 'LBP':
        for img in images:
            i = utils.load_image(folder + img)
            feature_vectors.append(get_LBP(i))
    elif feature_model == 'CM':
        for img in images:
            i = utils.load_image(folder + img)
            feature_vectors.append(get_CM(i))
    return np.array(feature_vectors), images

def get_labels(image_names, metadata_file):
    labels = []
    df = pd.read_csv(metadata_file)
    for image_name in image_names:
        try:
            image_row = df[df['imageName'] == image_name]
            labels.append(image_row['aspectOfHand'].values[0].split(' ')[0])

        except ValueError as e:
            print(e)
            sys.exit()

    return labels


def compute_stats(window):
    stats = []
    parts = cv2.split(window)

    for channel in parts:
        c = np.reshape(channel, -1)
        stats.append(np.mean(c))
        stats.append(np.std(c))
        stats.append(skew(c))

    return stats

def get_CM(img):
    output = []
    img_yuv = utils.convert_toYUV(img)

    for window_coords in utils.generate_blocks(img_yuv, 100, 100):
        x, y = window_coords[0], window_coords[1]
        window = img_yuv[x:x+100, y:y+100]
        stats = compute_stats(window)
        stats = np.array(stats).flatten()
        output.append(stats)

    output = np.array(output).flatten()

    return output

# Extracts the LBP from the image
# img is an image that has already been loaded by cv2
def get_LBP(img):
    output = []

    img_yuv = utils.convert_toYUV(img)

    for window_coords in utils.generate_blocks(img_yuv, 100, 100):
        x, y = window_coords[0], window_coords[1]
        window = img_yuv[x:x+100, y:y+100]
        histogram = [h for h in lbp(window[0], 1, 8)]
        output.append(histogram)

    output = np.array(output).flatten()

    return output

def get_CMLBP(image):
    output = np.append(get_CM(image), get_LBP(image))
    return output

def get_CM_range():
    maxes = [255.0, 117.172325, 71.55665, 151.993, 23.657257, 99.985, 231.8959, 57.69578, 99.985]
    mins = [0.0, 0.0, -99.985, 0.0, 0.0, -99.985, 0.0, 0.0, -99.985]
    num = 9
    return random_vector.ValueRanges(num, maxes, mins)

def create_LSH_index(LSH, vectors, image_names):
    index = {}

    for i in range(len(vectors)):
        hash = LSH.hash(vectors[i])
        for h in hash:
            if h not in index:
                index[h] = []
            index[h].append(image_names[i])

    # for item in index:
    #     print(f'index[{item}] = {index[item]}')
    return index


def get_similar_from_LSH(LSH, query_image, index, t, imageNames, dataMatrix, distance):
    hashed = LSH.hash(query_image)

    # the buckets returned from the hash
    hashed = list(set(hashed))
    # print("Buckets returned from query hash")
    # print(hashed)
# 
    bucket_values = []

    # create the index
    # print("Index values")
    for h in hashed:
        # print(index[h])
        bucket_values += index[h]

    total_images_considered = len(bucket_values)


    bucket_values = list(set(bucket_values))
    unique_images_considered = len(bucket_values)
    # print("Bucket Values")
    # print(bucket_values)
    # print(f"Length of Bucket values: {len(bucket_values)}")
    # sys.exit(-1)

    if len(bucket_values) < t:
        print("NEED MORE BUCKET VALUES")

    # image_names 
    # features = get_features(bucket_values, namesMatrix, dataMatrix)

    scores = {}

    for i in range(len(bucket_values)):
        name = imageNames[i]
        feature_np = dataMatrix[i]
        length = int(len(query_image - feature_np)) - 1
        score = 0
        if distance == 'manhatten':
            score = np.float32(np.sum(np.absolute(query_image - feature_np)))
        elif distance == 'euclidean':
            score = np.sqrt(np.float32(np.sum(np.square(query_image - feature_np))))
        else:
            print("Unknown distance measure")
            sys.exit(-1)
        # for i in range(num_comparisons):
        #     if distance == 'manhatten':
        #         score += np.float32(np.sum(np.absolute(query_image[:int(length/num_comparisons*(i+1))]-feature_np[:int(length/num_comparisons*(i+1))])))
        #     elif distance == 'euclidean':
        #         score += np.sqrt(np.float32(np.sum(np.square(query_image[:int(length/num_comparisons*(i+1))]-feature_np[:int(length/num_comparisons*(i+1))]))))

        scores[name] = score

    # for feature in features:
    #     name = feature[0]
    #     feature_np = np.float32(np.asarray(feature[6:]))
    #     length = int(len(query_image-feature_np)) - 1
    #     score = 0
    #     num_comparisons = 1
    #     for i in range(num_comparisons):
    #         if distance == 'manhatten':
    #             score += np.float32(np.sum(np.absolute(query_image[:int(length/num_comparisons*(i+1))]-feature_np[:int(length/num_comparisons*(i+1))])))
    #         elif distance == 'euclidean':
    #             score += np.sqrt(np.float32(np.sum(np.square(query_image[:int(length/num_comparisons*(i+1))]-feature_np[:int(length/num_comparisons*(i+1))]))))

    #     scores[name] = score
        
    l = []
    for key, value in scores.items():
        l.append((key, value))
    # print(l)

    s = sorted(l, key=lambda tup: tup[1])

    print(f"Total images considered: {total_images_considered}")
    print(f"Total unique images considered: {unique_images_considered}")

    
    return s[:t]


def compare_to_all(query_image, dataMatrix, namesMatrix):
    scores = []

    for i in range(len(dataMatrix)):
        scores.append((namesMatrix[i], np.sqrt(np.float32(np.sum(np.square(query_image - dataMatrix[i]))))))

    return scores
    

    

def get_range(featureMatrix):
    length = len(featureMatrix[0])
    minimums = np.zeros(length)
    maximums = np.zeros(length)

    for item in featureMatrix:
        for i in range (length):
            if item[i] > maximums[i]:
                maximums[i] = item[i]
            if item[i] < minimums[i]:
                minimums[i] = item[i]


    # for item in feature:
    #     if item > maximum:
    #         maximum = item
    #     if item < minimum:
    #         minimum = item

    # print('Minimums')
    # print(minimums[:18])
    # print()
    # print('Maximums')
    # print(maximums[:18])
    return random_vector.ValueRanges(length, maximums, minimums)


def prune_irrelevant(rfs_type, data, image_names, relevant_images, irrelevant_images):
    training_names = column(relevant_images, 0)
    training_names.extend(column(irrelevant_images, 0))

    relevant_images = np.array(column(relevant_images, 1))
    irrelevant_images = np.array(column(irrelevant_images, 1))

    relevant_images = np.c_[relevant_images, np.ones(len(relevant_images))]
    irrelevant_images = np.c_[irrelevant_images, np.zeros(len(irrelevant_images))]

    #begin to train the model
    if rfs_type == 'DTREE':
        model = dtree(max_depth=None, entropy_threshold=0.001)
    elif rfs_type == 'SVM':
        model = SVM(kernel='poly', kernel_param = 3, soft_margin = None)
    else:
        model = PPR(k=5, image_names=training_names)
        model.test_names = image_names

    r_and_i_images = np.concatenate((relevant_images, irrelevant_images), axis=0)
    np.random.shuffle(r_and_i_images)


    x_train = r_and_i_images[:,:-1]
    y_train = r_and_i_images[:,-1]
    model.fit(x_train, y_train)

    if rfs_type == 'PPR':
        y_pred = model.predict_t6(data)
    else:
        y_pred = model.predict(data)

    y_pred = np.array(y_pred)
    print("----Pruned {} irrelevant images----".format(len(y_pred[y_pred == 0])))
    
    relevant_indices = np.where(y_pred == 1)
    return data[relevant_indices], image_names[relevant_indices]

def modify_query(rfs_type, data, image_names, relevant_images, irrelevant_images, query_image_fv):
    training_names = column(relevant_images, 0)
    training_names.extend(column(irrelevant_images, 0))

    relevant_images = np.array(column(relevant_images, 1))
    irrelevant_images = np.array(column(irrelevant_images, 1))

    relevant_images = np.c_[relevant_images, np.ones(len(relevant_images))]
    irrelevant_images = np.c_[irrelevant_images, np.zeros(len(irrelevant_images))]

    #begin to train the model
    if rfs_type == 'DTREE':
        model = dtree(max_depth=None, entropy_threshold=0.001)
    elif rfs_type == 'SVM':
        model = SVM(kernel='poly', kernel_param = 3, soft_margin = None)
    else:
        model = PPR(k=5, image_names=training_names)
        model.test_names = image_names
        
    r_and_i_images = np.concatenate((relevant_images, irrelevant_images), axis=0)
    np.random.shuffle(r_and_i_images)


    x_train = r_and_i_images[:,:-1]
    y_train = r_and_i_images[:,-1]
    model.fit(x_train, y_train)

    if rfs_type == 'PPR':
        y_pred = model.predict_t6(data)
    else:
        y_pred = model.predict(data)

    y_pred = np.array(y_pred)
    
    relevant_indices = np.where(y_pred == 1)
    irrelevant_indices = np.where(y_pred == 0)
    relevant_data = data[relevant_indices]
    irrelevant_data = data[irrelevant_indices]
    if len(relevant_data) == 0:
        relevant_mean = np.array(column(relevant_images, 0)).mean(axis=0)
    else:
        relevant_mean = relevant_data.mean(axis=0)
    if len(irrelevant_data) == 0:
        irrelevant_mean = np.array(column(irrelevant_images, 0)).mean(axis=0)
    else:
        irrelevant_mean = irrelevant_data.mean(axis=0)
    difference_of_means = relevant_mean - irrelevant_mean

    #query_image_fv = (query_image_fv * 0.6) + (relevant_mean * 0.2) + (relevant_images[:,:-1].mean(axis=0) * 0.2)
    query_image_fv += difference_of_means
    print(query_image_fv)
    print(query_image_fv.shape)
    return query_image_fv
 
    

def get_indices_of_images(entire_image_names, names_to_find_indices_for):
    indicies = []
    for i in names_to_find_indices_for:
        indicies.append(np.where(entire_image_names == i)[0][0])
    return np.array(indicies)

def takeSecond(elem):
    return elem[1]

def reorder_t_results_probabilisticly(results, data, image_names, relevant_images, irrelevant_images):
    #TODO: implement
    indicies_of_results = get_indices_of_images(image_names, column(results,0))
    indicies_of_relevant = get_indices_of_images(image_names, column(relevant_images, 0))
    indicies_of_irrelevant = get_indices_of_images(image_names, column(irrelevant_images, 0))

    #convert everything to binary feature vectors
    medians = np.median(data, axis=0)
    binary_data = []
    for p in data:
        binary_data.append(np.where(p - medians >= 0, 1, 0))
    binary_data = np.array(binary_data)

    relevant_data = binary_data[indicies_of_relevant]
    irrelevant_data = binary_data[indicies_of_irrelevant]
    result_data = binary_data[indicies_of_results]

    n_i = np.sum(binary_data, axis=0)
    N = len(data)
    r_i = np.sum(relevant_data, axis=0)
    R = len(relevant_data)

    p_i = (r_i + (n_i / N)) / (R + 1)
    u_i = (n_i - r_i + (n_i / N)) / (N - R + 1)

    #l = r_i / (R - r_i)
    #r = (n_i - r_i) / (N - R - n_i + r_i)

    multiplier = np.log2((p_i * (1 - u_i))/(u_i * (1 - p_i)))
    #multiplier = np.log(l / r)

    #get probability that a feature vector is as such given relevant/irrelevant
    new_results = []
    for i in range(len(results)):
        new_results.append([results[i][0], np.sum(result_data[i] * multiplier)])

    new_results.sort(key=takeSecond, reverse=True)
    return new_results
