import os
import sys
import glob
import csv

# cv2, numpy, scipy, and mahotas need to be installed
import cv2
import numpy as np
from scipy.stats import skew
from mahotas.features.lbp import lbp
from skimage.feature import hog
from skimage.measure import block_reduce
from sklearn.decomposition import NMF, PCA, LatentDirichletAllocation, TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

def load_image_rgb(image_str):
    image = cv2.imread(image_str)
    if image is None:
        print(f'Error: Image doesn\'t exist -- {image_str}')
        sys.exit(-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_image(image_str):
    image = cv2.imread(image_str)
    return image

def convert_toYUV(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return img_yuv

def convert_toGray(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray

def generate_blocks(img, x_size, y_size):
    for i in range(0, len(img), x_size):
        for j in range(0, len(img[0]), y_size):
            yield [i, j]

def compute_stats(window):
    # output: <Y_mean, Y_std, Y_skew, U_mean, U_std, U_skew, V_mean, V_std, V_skew>
    stats = []
    parts = cv2.split(window)

    for channel in parts:
        c = np.reshape(channel, -1)
        stats.append(str(np.mean(c)))
        stats.append(str(np.std(c)))
        stats.append(str(skew(c)))

    return stats

# This function extracts the Color Moments from an image 
# img is an image that has already been loaded by cv2
def get_CM(img):
    # output = ['Y_mean, Y_std, Y_skew, U_mean, U_std, U_skew, V_mean, V_std, V_skew']
    output = []
    img_yuv = convert_toYUV(img)

    for window_coords in generate_blocks(img_yuv, 100, 100):
        x, y = window_coords[0], window_coords[1]
        window = img_yuv[x:x+100, y:y+100]
        stats = compute_stats(window)
        # stats = [str(stat) for stat in compute_stats(window)]
        line = ','.join(stats)
        output.append(line)

    return output

# Extracts the LBP from the image
# img is an image that has already been loaded by cv2
def get_LBP(img):
    output = []

    img_yuv = convert_toYUV(img)

    for window_coords in generate_blocks(img_yuv, 100, 100):
        x, y = window_coords[0], window_coords[1]
        window = img_yuv[x:x+100, y:y+100]
        histogram = [str(h) for h in lbp(window[0], 1, 8)]
        line = ','.join(histogram)
        output.append(line)

    return output

#gets the feature vector for HOG
def get_HOG(img):
    img = convert_toGray(img)
    img = block_reduce(img, block_size=(10, 10), func=np.mean)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys', cells_per_block=(2, 2), visualize=True, feature_vector=True)
    #plt.imshow(hog_image, cmap=plt.cm.gray) 
    #plt.show()
    return fd

#returns a flattened feature vector of size K x 128 where
#K is the number of keypoints found on the image
#the 128 elements are the histogram of oriented gradients
def get_SIFT(image):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)

    return des.flatten()

def get_n_SIFT(image, n):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n)
    kp, des = sift.detectAndCompute(image, None)

    #idk why this happens...
    if(len(des) > n):
        des = des[0:n]

    return des.flatten()

# for an image name, i.e. 'Hands_0000002.jpg', it finds the tags we want and outputs them
# info_file is the url of the HandInfo.csv file
def get_tags(image_name, info_file):
    found_row = ''
    output = []
    with open(info_file) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[7] == image_name:
                found_row = row
                break

    if found_row == '':
        print('Error -- Image ("{}") not found'.format(image_name))
        sys.exit(-1)

    output += found_row[6].split()[::-1]
    output += 'accessories'.split() if found_row[4] == '1' else 'noAccessories'.split()
    output += found_row[2].split()
    output += found_row[0].split()

    return output

# image_str -- url to the image
# feature -- 'CM' or 'LBP'
# output_filename -- the csv file you want to use 
#       (doesn't need to be created, if it already exists it will append to the existing one)
# info_file -- the HandInfo.csv file url
def output_to_file(image_str, feature, output_filename, info_file):    
    if feature != 'CM' and feature != 'LBP' and feature !='HOG' and feature != 'SIFT' and feature != 'SIFT':
        print('Error: Unrecognized feature requested:', feature)
        sys.exit(-1)

    img = load_image(image_str)

    if feature == 'CM':
        output = get_CM(img)
    elif feature == 'LBP':
        output = get_LBP(img)
    elif feature == 'HOG':
        output = get_HOG(img)
    elif feature == 'SIFT' or feature =='SIFT':
        output = get_SIFT(img)

    image_name = image_str.split('/')[-1]

    tags = get_tags(image_name, info_file)

    output = "{},{},{}".format(image_name, ','.join(tags), ','.join(map(str, output)))

    with open(output_filename, 'a') as f:
        f.write(output)
        f.write('\n')

# folder -- the image database folder url 
# output_filename -- the csv file you want to output to
# feature_model -- 'CM' or 'LBP'
# info_file -- the HandInfo.csv file url
def process_images(folder, output_filename, feature_model, info_file):
    filecounter = len(glob.glob1(folder,"*.jpg"))
    print('Processing Images -- {} to process'.format(filecounter))
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            filecounter -= 1            
            print('{} -- {} to go'.format(filename, filecounter))            
            output_to_file(os.path.join(folder, filename), feature_model, output_filename, info_file)
        else:
            continue




#BEGIN PROJECT 2 FUNCTIONS

#Normalize CM for NMF and LDA only
def cm_min_max(dataMatrix):
    minVal = [0] * 3
    maxVal = [0] * 3
    numWindows = int(len(dataMatrix[0]) / 3)
    meanColumns = np.arange(numWindows) * 3
    varColumns = (np.arange(numWindows) * 3) + 1
    skewColumns = (np.arange(numWindows) * 3) + 2
    meanMatrix = dataMatrix[:, meanColumns]
    varMatrix = dataMatrix[:, varColumns]
    skewMatrix = dataMatrix[:, skewColumns]
    return [[np.min(meanMatrix), np.min(varMatrix), np.min(skewMatrix)], [np.max(meanMatrix), np.max(varMatrix), np.max(skewMatrix)]]


def normalize_cm(dataMatrix, minMax, matrixWise=True):
    if matrixWise:
        normalizedMatrix = dataMatrix
        for i in range(len(dataMatrix)):
            for j in range(len(dataMatrix[i])):
                if j % 3 == 0:
                    normalizedMatrix[i][j] = (dataMatrix[i][j] - minMax[0][0]) / (minMax[1][0] - minMax[0][0])
                if j % 3 == 1:
                    normalizedMatrix[i][j] = (dataMatrix[i][j] - minMax[0][1]) / (minMax[1][1] - minMax[0][1])
                if j % 3 == 2:
                    normalizedMatrix[i][j] = (dataMatrix[i][j] - minMax[0][2]) / (minMax[1][2] - minMax[0][2])

        return normalizedMatrix
    else:
        normalizedMatrix = dataMatrix
        for j in range(len(dataMatrix)):
            if j % 3 == 0:
                normalizedMatrix[j] = (dataMatrix[j] - minMax[0][0]) / (minMax[1][0] - minMax[0][0])
            if j % 3 == 1:
                normalizedMatrix[j] = (dataMatrix[j] - minMax[0][1]) / (minMax[1][1] - minMax[0][1])
            if j % 3 == 2:
                normalizedMatrix[j] = (dataMatrix[j] - minMax[0][2]) / (minMax[1][2] - minMax[0][2])

        return normalizedMatrix


#function to get the sift data 'matrix' 
#isn't rectangular so can't use pandas to read (sad face)
def get_sift_data_matrix(fileName, label=None):
    dataMatrix = []
    if label != None:
        index = get_label_index(label)
        with open(fileName, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row[index] == label:
                    dataMatrix.append(np.float32(np.asarray(row[6:])))
    else:
        with open(fileName, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                dataMatrix.append(np.float32(np.asarray(row[6:])))

    return dataMatrix

def get_label_index(label):
    if label == 'left' or label == 'right':
        return 1
    elif label == 'palmar' or label == 'dorsal':
        return 2
    elif label == 'noAccessories' or label=='accessories':
        return 3
    elif label == 'male' or label == 'female':
        return 4
    elif type:
        return 5
    else:
        print('Error: Unrecognized label requested:', label)
        sys.exit(-1)

#function to return the data matrix,
#can identify a specific label to return, default is all
def get_data_matrix(fileName, feature_model, label=None): 
    #sift is handled differently
    if feature_model == 'SIFT':
        return get_sift_data_matrix(fileName, label)

    df = pd.read_csv(fileName, header=None)
    dataMatrix = []
    if label != None:
        index = get_label_index(label)

        if index == 5:
            for row in df.values:
                if row[index] == int(label):
                    dataMatrix.append(row[6:])
        else:
            #read in only those with the corresponding label
            for row in df.values:
                if row[index] == label:
                    dataMatrix.append(row[6:])

        return np.float32(np.asarray(dataMatrix))       
    else:
        #read in all
        dataMatrix = df.values[:, 6:]
        return np.float32(np.asarray(dataMatrix))

def apply_weights(dataMatrix, weightMatrix, feature_model, matrix_wise=True):
    if feature_model == 'SIFT':
        sift_matrix = []
        if not matrix_wise:
            #only dealing with one image so shouldn't return a list of images
            numKeypoints = int(len(dataMatrix) / 128)
            image_vector = np.reshape(dataMatrix, (numKeypoints, 128))
            weighted_image_matrix = image_vector.dot(weightMatrix.T).flatten()
            return weighted_image_matrix
        for image_vector in dataMatrix:
            numKeypoints = int(len(image_vector) / 128)
            image_vector = np.reshape(image_vector, (numKeypoints, 128))
            weighted_image_matrix = image_vector.dot(weightMatrix.T)
            sift_matrix.append(np.array(weighted_image_matrix.flatten()))
        return sift_matrix
    else:
        return dataMatrix.dot(weightMatrix.T)

def get_individual_feature_vector(CSVfileName, image_name, feature_model):
    if feature_model == 'SIFT':
        with open(CSVfileName, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row[0] == image_name:
                    return np.float32(np.asarray(row[6:]))
    else:
        df = pd.read_csv(CSVfileName, header=None)
        for row in df.values:
            if row[0] == image_name:
                return np.float32(np.array(row[6:]))

def get_image_names(CSVfileName, feature_model, label=None, subjectID=None):
    if feature_model == 'SIFT':
        image_names = []
        with open(CSVfileName, 'r') as csv_file:
            reader = csv.reader(csv_file)
            if label == None:
                for row in reader:
                    image_names.append(row[0])
            else:
                index = get_label_index(label)
                for row in reader:
                    if row[index] == label:
                        image_names.append(row[0])
        return np.asarray(image_names)
    else:
        df = pd.read_csv(CSVfileName, header=None)
        if subjectID == None and label == None:
            return df.values[:,0]
        elif label == None:
            return df.values[df.values[:,5] == subjectID,0]
        elif subjectID == None:
            index = get_label_index(label)
            return df.values[df.values[:,index] == label,0]
        else:
            index = get_label_index(label)
            d = df.values[df.values[:,5] == subjectID, :7]
            return d[d[:,index] == label, 0]

def get_image_names_and_labels(CSVfileName, feature_model, label=None, subjectID=None):
    if feature_model == 'SIFT':
        image_names = []
        with open(CSVfileName, 'r') as csv_file:
            reader = csv.reader(csv_file)
            if label == None:
                for row in reader:
                    image_names.append(row[0:6])
            else:
                index = get_label_index(label)
                for row in reader:
                    if row[index] == label:
                        image_names.append(row[0:6])
        return np.asarray(image_names)
    else:
        df = pd.read_csv(CSVfileName, header=None)
        if subjectID == None and label == None:
            return df.values[:,0:6]
        elif label == None:
            return df.values[df.values[:,5] == subjectID,0:6]
        elif subjectID == None:
            index = get_label_index(label)
            return df.values[df.values[:,index] == label,0:6]
        else:
            index = get_label_index(label)
            d = df.values[df.values[:,5] == subjectID, 0:6]
            return d[d[:,index] == label]

def get_closest_images(dataMatrix, image_vector, m, feature_model, k):
    if feature_model == 'SIFT':
        distances = sift_distances(dataMatrix, image_vector, k)
    else:
        dataMatrix = np.asarray(dataMatrix, dtype=np.float32)
        image_vector = np.asarray(image_vector, dtype=np.float32)
        distances = euclidean_distance_mv(dataMatrix, image_vector)

    #sorts the array and selectes the top m indexes
    ind = np.argpartition(distances, m+1)[:m+1]
    ind = ind[np.argsort(distances[ind])]

    #if chosen image is in data still
    #prune it from list here
    if(distances[ind[0]] < 5e-3):
        ind = ind[1:]
    else:
        ind = ind[:-1]

    return ind, distances

#calculates the estimated similarity for a given label, does so by getting that labels
#latent semantics and finding the cosine similarity with it, and weighting each cosine similarity
#with the corresponding eigenvalues, so similarity on a more meaningful vector is considered better
def get_label_similarity_estimate(csv_file, k, technique, feature_model, query_image_feature_vector, label):
    if(feature_model == 'SIFT'):
        dataMatrix = get_data_matrix(csv_file, feature_model, label=label)
        weightMatrix, _ = extract_k_latent_semantics(dataMatrix, k, technique, feature_model)
        if(technique == 'SVD' or technique == 'PCA'):
            weightImportance = get_ls_importances(reshape_sift_matrix(dataMatrix), k, technique, feature_model)
        else:
            weightImportance = [1] * k
        
        similarityEstimate = 0
        keypointMatrix = query_image_feature_vector.reshape((int(len(query_image_feature_vector)/128), 128))
        for i in range(len(weightMatrix)):
            keypointSimTotal = 0
            for keypoint in keypointMatrix:
                keypointSimTotal += cosine_similarity_vv(keypoint, weightMatrix[i])
            keypointAvg = keypointSimTotal / len(keypointMatrix)
            similarityEstimate += keypointAvg * weightImportance[i]

        return similarityEstimate
    else:
        dataMatrix = get_data_matrix(csv_file, feature_model, label=label)
        weightMatrix, _ = extract_k_latent_semantics(dataMatrix, k, technique, feature_model)
        if(technique == 'SVD' or technique == 'PCA'):
            weightImportance = get_ls_importances(dataMatrix, k, technique, feature_model)
        else:
            weightImportance = [1] * k
        
        similarityEstimate = 0
        for i in range(len(weightMatrix)):
            similarityEstimate += cosine_similarity_vv(query_image_feature_vector, weightMatrix[i]) * weightImportance[i]

        return similarityEstimate

#calculates the estimated distance for a given label, does so by getting that labels
#latent semantics and applying those weights then getting the average distance between
#the query image and all images with the specified 
def get_average_distance(dataMatrix, k, feature_model, query_image_feature_vector):
    if(feature_model != 'SIFT'):
        distances = euclidean_distance_mv(dataMatrix, query_image_feature_vector)
        #euclidean distance seems to have better performance on the sample dataset based on results:
        #parameters: feature_model='LBP'; technique='LDA'; k=5; feature_label='left'
        #error count: eucidean:25; cosine:75; manhatten:37
        # for i in range(len(dataMatrix)):
        #     distances =[]+[cosine_similarity_vv(dataMatrix[i], query_image_feature_vector)]
        return np.mean(distances)
    else:
        distances = sift_distances(dataMatrix, query_image_feature_vector, k)
        return np.mean(distances)


#each vote is not weighted
#returns list of votes, example: [[2,5],[1,6],[7,0],[3,4]]
def knn_default(ind, dists, imageDataMatrix):
    votes = [[0,0],[0,0],[0,0],[0,0]]
    for i in range(len(ind)):
        nearbyNeighbor = imageDataMatrix[ind[i]]
        if nearbyNeighbor[1] == 'right':
            votes[0][0] += 1
        else:
            votes[0][1] += 1
        if nearbyNeighbor[2] == 'dorsal':
            votes[1][0] += 1
        else:
            votes[1][1] += 1
        if nearbyNeighbor[3] == 'accessories':
            votes[2][0] += 1
        else:
            votes[2][1] += 1
        if nearbyNeighbor[4] == 'male':
            votes[3][0] += 1
        else:
            votes[3][1] += 1
    return votes


#each vote is weighted by 1/distance^2
#returns list of votes, example: [[2,5],[1,6],[7,0],[3,4]]
def knn_distance_weights(ind, dists, imageDataMatrix):
    votes = [[0,0],[0,0],[0,0],[0,0]]
    for i in range(len(ind)):
        nearbyNeighbor = imageDataMatrix[ind[i]]
        if nearbyNeighbor[1] == 'right':
            votes[0][0] += 1.0 / (dists[i]**2)
        else:
            votes[0][1] += 1.0 / (dists[i]**2)
        if nearbyNeighbor[2] == 'dorsal':
            votes[1][0] += 1.0 / (dists[i]**2)
        else:
            votes[1][1] += 1.0 / (dists[i]**2)
        if nearbyNeighbor[3] == 'accessories':
            votes[2][0] += 1.0 / (dists[i]**2)
        else:
            votes[2][1] += 1.0 / (dists[i]**2)
        if nearbyNeighbor[4] == 'male':
            votes[3][0] += 1.0 / (dists[i]**2)
        else:
            votes[3][1] += 1.0 / (dists[i]**2)
    return votes

#each vote is weighted by the inverse of the frequencies
def knn_frequency_weights(ind, dists, imageDataMatrix):
    votes = [[0,0],[0,0],[0,0],[0,0]]
    length = len(imageDataMatrix)

    #get proportions of all labels
    propRight = len(imageDataMatrix[imageDataMatrix[:,1] == 'right'])/length
    propLeft = 1 - propRight
    propDorsal = len(imageDataMatrix[imageDataMatrix[:,2] == 'dorsal'])/length
    propPalmar = 1 - propDorsal
    propAccessories = len(imageDataMatrix[imageDataMatrix[:,3] == 'accessories'])/length
    propNoAccessories = 1 - propAccessories
    propMale = len(imageDataMatrix[imageDataMatrix[:,4] == 'male'])/length
    propFemale = 1 - propMale

    for i in range(len(ind)):
        nearbyNeighbor = imageDataMatrix[ind[i]]
        if nearbyNeighbor[1] == 'right':
            votes[0][0] += 1.0 / propRight
        else:
            votes[0][1] += 1.0 / propLeft
        if nearbyNeighbor[2] == 'dorsal':
            votes[1][0] += 1.0 / propDorsal
        else:
            votes[1][1] += 1.0 / propPalmar
        if nearbyNeighbor[3] == 'accessories':
            votes[2][0] += 1.0 / propAccessories
        else:
            votes[2][1] += 1.0 / propNoAccessories
        if nearbyNeighbor[4] == 'male':
            votes[3][0] += 1.0 / propMale
        else:
            votes[3][1] += 1.0 / propFemale
    return votes

#classify the vote vector and give back string results with
#percentages
def classify_votes(votes):
    classifications = []
    percentages = []
    
    if votes[0][0] > votes[0][1]:
        classifications.append('right')
        percentages.append(votes[0][0] / (votes[0][0] + votes[0][1]))
    else:
        classifications.append('left')
        percentages.append(votes[0][1] / (votes[0][0] + votes[0][1]))
    if votes[1][0] > votes[1][1]:
        classifications.append('dorsal')
        percentages.append(votes[1][0] / (votes[1][0] + votes[1][1]))
    else:
        classifications.append('palmar')
        percentages.append(votes[1][1] / (votes[1][0] + votes[1][1]))
    if votes[2][0] > votes[2][1]:
        classifications.append('accessories')
        percentages.append(votes[2][0] / (votes[2][0] + votes[2][1]))
    else:
        classifications.append('noAccessories')
        percentages.append(votes[2][1] / (votes[2][0] + votes[2][1]))
    if votes[3][0] > votes[3][1]:
        classifications.append('male')
        percentages.append(votes[3][0] / (votes[3][0] + votes[3][1]))
    else:
        classifications.append('female')
        percentages.append(votes[3][1] / (votes[3][0] + votes[3][1]))

    return classifications, percentages



def extract_k_latent_semantics(dataMatrix, k, technique, feature_model):
    if technique != 'PCA' and technique != 'SVD' and technique !='LDA' and technique != 'NMF':
        print('Error: Unrecognized technique requested:', technique)
        sys.exit(-1)

    #convert SIFT into uniform matrix of size q by 128 where
    #q is the total number of keypoints
    if feature_model == 'SIFT':
        dataMatrix = reshape_sift_matrix(dataMatrix)

    if technique == 'PCA':
        feature_latent_semantics, data_latent_semantics = get_PCA(dataMatrix, k)
    elif technique == 'SVD':
        feature_latent_semantics, data_latent_semantics = get_SVD(dataMatrix, k)
    elif technique == 'LDA':
        feature_latent_semantics, data_latent_semantics = get_LDA(dataMatrix, k)
    else:
        feature_latent_semantics, data_latent_semantics = get_NMF(dataMatrix, k)

    return np.asarray(feature_latent_semantics), np.asarray(data_latent_semantics)

def sort_into_term_weight_pairs(latentMatrix):
    orderedtermweightpairsmatrix = []
    for i in range(len(latentMatrix)):
        orderedtermweightpairs = [(index, value) for index, value in sorted(enumerate(latentMatrix[i]), reverse=True, key=lambda x: x[1])]
        orderedtermweightpairsmatrix.append(orderedtermweightpairs)

    return orderedtermweightpairsmatrix
        
#displays the (index, weight) of every element in every extracted feature
def display_term_weight_pairs(fls, dls, image_names, feature_model):
    orderedfls = sort_into_term_weight_pairs(fls)
    ordereddls = sort_into_term_weight_pairs(dls)

    #display the feature latent semantic vector
    print("FEATURE LATENT SEMANTICS:")
    for weight_vector in orderedfls:
        print(weight_vector)
        print('\n')

    if feature_model != 'SIFT':
        #display the data latent semantic vector
        dls_with_inames = []
        for latent_semantic in ordereddls:
            withinames = []
            for tup in latent_semantic:
                withinames.append((image_names[tup[0]], tup[1]))
            dls_with_inames.append(withinames)
    else:
        #can't assign image names if SIFT
        #because the data is keypoints, not images
        dls_with_inames = ordereddls
    print("DATA LATENT SEMANTICS:")
    for weight_vector in dls_with_inames:
        print(weight_vector)
        print('\n')

#each latent semantic here is a list of images with their ranks
#sort them and get the highest ranked images to display
def display_dls(dm, dls, images, image_folder):
    ordereddls = sort_into_term_weight_pairs(dls)
    for ls in ordereddls:
        fig = plt.figure(figsize=(16,16))
        numRows = int(len(ls) / 5) + 1
        for i in range(len(ls)):
            imsp = fig.add_subplot(numRows, 5, i+1)
            imsp.imshow(load_image_rgb(os.path.join(image_folder,images[ls[i][0]])))
            imsp.text(1, 0.5, str(ls[i][1]))
            imsp.axes.get_xaxis().set_visible(False)
            imsp.axes.get_yaxis().set_visible(False)
        plt.show()

#each latent semantic here is the weighted values of each feature
#display the image most similar (highest dot product) for each 
#latent semantic
def display_fls(dm, fls, images, image_folder):
    for ls in fls:
        dots = dm.dot(ls.T)
        bestImageIndex = np.argmax(dots)
        plt.imshow(load_image_rgb(os.path.join(image_folder , images[bestImageIndex])))
        plt.title(images[bestImageIndex])
        plt.show()


def get_PCA(dataMatrix, k):
    #to be implemented
    pca = PCA(n_components=k)

    return pca.fit(dataMatrix).components_, pca.fit(dataMatrix.T).components_


def get_SVD(dataMatrix, k):
    svd = TruncatedSVD(n_components=k)
    
    return svd.fit(dataMatrix).components_, svd.fit(dataMatrix.T).components_

#percentage of variance/data perserved with the given number of latent semantics
#used to weight the value of each latent semantic 
#only applicable for SVD and PCA
def get_ls_importances(dataMatrix, k, technique, feature_model):
    if technique == 'PCA':
        pca = PCA(n_components=k)
        return pca.fit(dataMatrix).explained_variance_
    elif technique == 'SVD':
        svd = TruncatedSVD(n_components=k)
        return svd.fit(dataMatrix).explained_variance_
    return None

#Input should be the matrix from phase 1
#k = number of latent semantics to be produced
    
def get_LDA(dataMatrix, k):    
    #Create the model with the appropriate parameters
    model = LatentDirichletAllocation(n_components=k, random_state=0)

    return model.fit(dataMatrix).components_, model.fit(dataMatrix.T).components_

#Input should be the matrix from phase 1
#k = number of latent semantics to be produced
def get_NMF(dataMatrix, k):
    #Create the model with the appropriate parameters
    model = NMF(n_components=k, init='random', random_state=0)
    return model.fit(dataMatrix).components_, model.fit(dataMatrix.T).components_

#euclidean distance of matrix(m) and a vector(v)
#returns a vector of all the distances
def euclidean_distance_mv(dataMatrix, feature_vector):
    return np.sqrt(np.float32(np.sum(np.square(dataMatrix - feature_vector), axis=1)))

def manhatten_distance_mv(dataMatrix, feature_vector):
    return np.float32(np.sum(np.absolute(dataMatrix - feature_vector), axis=1))

def negative_cosine_similarity_mv(dataMatrix, feature_vector):
    #negative for now so that don't have to modify sorting algo
    return -1 * cosine_similarity_mv(dataMatrix, feature_vector)

def cosine_similarity_mv(dataMatrix, feature_vector):
    return np.asarray((np.divide(np.dot(dataMatrix, feature_vector.T),np.linalg.norm(dataMatrix, axis=1)*np.linalg.norm(feature_vector))), dtype=np.float32)

def cosine_similarity_vv(v1, v2):
    return 1 - distance.cosine(v1, v2)

#takes the messy sift matrix with variable length rows and modifies it
#to strictly have 128 columns and the rows are keypoints instead of images
#128 because that is the size of a feature vector for an individual keypoint
def reshape_sift_matrix(dataMatrix):
    reshaped_matrix = []
    for row in dataMatrix:
        numKeypoints = int(len(row) / 128)
        m = row.reshape((numKeypoints, 128))
        reshaped_matrix.extend(m)

    return np.float32(np.asarray(reshaped_matrix))

#############################
### Task 6 Stuff ############
#############################

def get_subjects(csv_file):     
    subjects = []

    df = pd.read_csv(csv_file, header=None)

    index = 5

    for row in df.values:
        if row[5] not in subjects:
            subjects.append(row[index])

    return subjects

def get_m_similar_subjects(subjectID, m, k, CSV_file, technique, feature_model):
    # get all the subjects in the CM file
    subjects = get_subjects(CSV_file)

    latent, _ = extract_k_latent_semantics(get_data_matrix(CSV_file, feature_model), k, technique, feature_model)

    # average the latent semantics of each image for
    # each subject, to obtain a single vector for each
    # subject in the meta data file
    subject_vectors = []
    for subject in subjects:
        dataMatrix = get_data_matrix(CSV_file, feature_model, label=subject)

        dataMatrix = apply_weights(dataMatrix, latent, feature_model)

        subject_vectors.append(np.mean(dataMatrix, axis=0))

    # get the index of the subject
    try:
        subjectID_index = subjects.index(subjectID)
    except:
        subjectID_index = -1
        print("subject does not exist")

    distances = manhatten_distance_mv(subject_vectors, subject_vectors[int(subjectID_index)])

    ind = np.argpartition(distances, 3+1)[:3+1]
    ind = ind[np.argsort(distances[ind])]

    #if chosen subject is in data still
    #prune it from list here
    if(distances[ind[0]] < 5e-5):
        ind = ind[1:]
    else:
        ind = ind[:-1]

    # ind refers to the most similar subjects
    # distances are the distances of each subject
    # subjects are the subjects in the metadata file, 
    # and correspond to the distance at the same index
    # of the distances array
    # i.e. the distance of subject[i] is at index i in 
    # distances, so distances[i]
    return ind, distances, subjects

def display_subject_images(subjectID, dist, csv_file, image_folder, feature_model):
    image_names = get_image_names(csv_file, feature_model, subjectID=subjectID)
    
    fig = plt.figure(figsize=(16,16))
    numRows = int(len(image_names) / 5) + 2
    titlesp = fig.add_subplot(numRows, 1, 1)
    titlesp.text(0.5,0.5, "Subject {} has distance {}".format(str(subjectID), str(dist)))
    titlesp.axes.get_xaxis().set_visible(False)
    titlesp.axes.get_yaxis().set_visible(False)

    for i in range(len(image_names)):
        imsp = fig.add_subplot(numRows, 5, i+6)
        imsp.imshow(load_image_rgb(os.path.join(image_folder,image_names[i])))
        imsp.axes.get_xaxis().set_visible(False)
        imsp.axes.get_yaxis().set_visible(False)
    plt.show()
###############################

#####################
#TASK 7
#####################
#doesn't work for sift
def get_subject_ids_for_all_images(CSV_file):
    df = pd.read_csv(CSV_file, header=None)
    return np.array([df.values[:,5]])

def get_subjects_average_feature_vectors(CSV_file, feature_model, subjects):
    dataMatrix = get_data_matrix(CSV_file, feature_model)
    subIDS = get_subject_ids_for_all_images(CSV_file)
    dataMatrix = np.append(subIDS.T, dataMatrix, axis=1)
    subAvgFVDict = {}

    #loop through all subjects and get their means
    for sub in subjects:
        subAvgFVDict[str(sub)] = np.array(dataMatrix[dataMatrix[:,0] == sub,1:].mean(axis=0), dtype=np.float32)

    return subAvgFVDict

###############
#TASK 8
###############
def get_binary_metadata_matrix(CSV_file):
    output = []
    metadataAttributes = ['male', 'fairSkin', 'darkSkin', 
                'mediumSkin', 'veryFairSkin', 'accessories', 
                'nailPolish', 'dorsal', 'right', 'irregularities']
    with open(CSV_file) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            output.append(parse_metadata_row(row))
    output = np.array(output)
    image_names = output[:,0]
    output = output[:,1:]
    output = np.array(output, dtype=np.float32)

    return output, image_names, metadataAttributes


#parse the row to put in binary format    
def parse_metadata_row(row):
    out = []
    out.append(row[7])
    if row[2] == 'male':
        out.append(1)
    else:
        out.append(0)
    out.extend(one_hot_skinColor(row[3]))
    out.append(int(row[4]))
    out.append(int(row[5]))
    dorsalRight = row[6].split()
    if dorsalRight[0] == 'dorsal':
        out.append(1)
    else:
        out.append(0)
    if dorsalRight[1] == 'right':
        out.append(1)
    else:
        out.append(0)
    out.append(int(row[8]))

    return np.array(out)



#one hot encode skin color, so that only one of the values will be 1
def one_hot_skinColor(skinColor):
    if skinColor == 'fair':
        return [1, 0, 0, 0]
    elif skinColor == 'dark':
        return [0, 1, 0, 0]
    elif skinColor == 'medium':
        return [0, 0, 1, 0]
    elif skinColor == 'very fair':
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0] #shouldn't ever happen



#SIFT stuff
#reorder the dataMatrix so that keypoints
#match as best as possible
#POOR RESULTS
#which is a bummer cuz this was difficult
def reorder_keypoints(dataMatrix, qfv):
    qfm = qfv.reshape((int(len(qfv)/128), 128))
    dataMatrixReordered = []
    for fv in dataMatrix:
        fm = fv.reshape((int(len(fv)/128), 128))
        fmreordered = [None] * int(len(fv)/128)
        distances = []
        for keypoint in qfm:
            d = euclidean_distance_mv(fm, keypoint)
            distances.append(d)
        distances = np.array(distances, dtype=np.float32)

        maxValue = np.amax(distances)
        minPs = []
        for i in range(len(qfm)):
            minElementPosition = np.where(distances == np.amin(distances))
            minElementPosition = (minElementPosition[0][0], minElementPosition[1][0])
            

            #the best current match will be stored into it's corresponding
            #slot in the reordered matrix
            fmreordered[minElementPosition[0]] = fm[minElementPosition[1]]

            #now we no longer need to find a keypoint match for the 
            #minElementPosition[0]th keypoint in the query image
            #so we can assign the whole row to the max value so it isn't selected again
            distances[minElementPosition[0],:] = maxValue+1
            #also, the minElementPosition[1]th keypoint in the dataset image
            #has been used and cannot be duplicated so assign the max value to
            #that column as well
            distances[:, minElementPosition[1]] = maxValue+1
            minPs.append(minElementPosition)
        Xvals = []
        Yvals = []
        for minP in minPs:
            if minP[0] in Xvals or minP[1] in Yvals:
                print(minP, sorted(Xvals))
                print(distances)
            if minP[0] not in Xvals:
                Xvals.append(minP[0])
            if minP[1] not in Yvals:
                Yvals.append(minP[1])
        for i in range(31):
            if fmreordered[i] is None:
                print(i)
        fmreordered = np.array(fmreordered, dtype=np.float32)
        dataMatrixReordered.append(fmreordered.flatten())
    
    dataMatrixReordered = np.array(dataMatrixReordered, dtype=np.float32)
    return dataMatrixReordered


#get closest keypoint matches
def getClosestMatches(kp, fd2, n):
    mtx = euclidean_distance_mv(fd2, kp)
    matches = []
    maximum = np.amax(mtx)
    for i in range(n):
        minimum = np.amin(mtx)
        minIndex = np.where(mtx == np.amin(mtx))[0][0]
        mtx[minIndex] = maximum     #this is so we can find the next lowest min
        matches.append([minIndex, minimum])
    return np.array(matches)

#returns an array where each element is a set of n matches, 
#each match having the list: [queryIdx, trainIdx, distance]
def getKeypointMatches(fd1, fd2, n):
    matches = []
    for i in range(len(fd1)):
        t = getClosestMatches(fd1[i], fd2, n) #t is a n x 2 matrix with trainIdx and distance as columns
        iArr = np.array([i] * n)
        t = np.c_[iArr, t]  #append a column of i to t so that the query index is included in the match
        #print(t[0][0], t[0][1], t[0][2])
        matches.append(t)       #[query index, train index, distance]
    return matches

#matchCountWeight is for how much to weigh the amount
#of matches, 1 to only care about number of matches, 0 to only care about match
#distance average
def compareTwoSIFT(fd1, fd2, k, threshold=0.8, matchCountWeight=0.5):
    if(matchCountWeight < 0) or (matchCountWeight > 1):
        matchCountWeight = 0

    #reshape the feature vectors into matrices
    fd1 = np.reshape(fd1, ((int(len(fd1) / k), k)))
    fd2 = np.reshape(fd2, ((int(len(fd2) / k), k)))

    # get match pairs between two images
    matches = getKeypointMatches(fd1, fd2, 2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m[2] < 0.8*n[2]:
            good.append([m[2]])

    good = np.array(good)

    if(len(good) == 0):
        return float('inf')

    distances = np.sqrt(np.square(good).sum()) #square them to punish extreme differences
    matchCountWeightUpdate = (1 / len(good)) + (((len(good)-1)/len(good))*matchCountWeight)
    distanceAvg = (distances / (len(good) * (len(good) * matchCountWeightUpdate)))

    return distanceAvg

def sift_distances(dataMatrix, query_image_vector, k):
    distances = []
    for image_vector in dataMatrix:
        distances.append(compareTwoSIFT(np.array(query_image_vector, dtype=np.float32), np.array(image_vector, dtype=np.float32), k))

    return np.asarray(distances)
