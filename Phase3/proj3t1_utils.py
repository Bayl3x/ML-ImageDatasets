import argparse
import sys
import os
import cv2
import pandas as pd
import numpy as np
import glob
from mahotas.features.lbp import lbp
from sklearn.decomposition import PCA
from scipy.spatial import distance
import csv
from scipy.stats import skew
import utils

def load_image(image_str):
    image = cv2.imread(image_str)
    return image

def convert_toYUV(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return img_yuv

def generate_blocks(img, x_size, y_size):
    for i in range(0, len(img), x_size):
        for j in range(0, len(img[0]), y_size):
            yield [i, j]


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

def get_LBP(img):
    output = []
    img=load_image(img)
    img_yuv = convert_toYUV(img)

    for window_coords in generate_blocks(img_yuv, 100, 100):
        x, y = window_coords[0], window_coords[1]
        window = img_yuv[x:x+100, y:y+100]
        histogram = [str(h) for h in lbp(window[0], 1, 8)]
        line = ','.join(histogram)
        output.append(line)

    return output

def get_data_matrix(folder, info_file):
    data_matrix_dorsal=[]
    data_matrix_palmar=[]
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        if get_label(img,info_file)=='dorsal':
            output=get_LBP(img_path)
            data_matrix_dorsal.append(','.join(map(str,output)))
        else:
            output=get_LBP(img_path)
            data_matrix_palmar.append(','.join(map(str,output)))
    #print(len(data_matrix_dorsal), len(data_matrix_palmar))
    data_matrix_dorsal=[[float(x) for x in line.split(',')] for line in data_matrix_dorsal]
    data_matrix_palmar=[[float(x) for x in line.split(',')] for line in data_matrix_palmar]
    return np.float32(np.asarray(data_matrix_dorsal)),np.float32(np.asarray(data_matrix_palmar))

def get_PCA(dataMatrix, k):
    pca = PCA(n_components=k)
    return pca.fit(dataMatrix).components_, pca.fit(dataMatrix.T).components_

def extract_k_latent_semantics(data_matrix, k):
    feature_latent_semantics, _ = get_PCA(data_matrix, k)
    return np.asarray(feature_latent_semantics)

def cosine_similarity(vector_a, vector_b):
    return 1-distance.cosine(vector_a,vector_b)

def get_label(image_name, info_file):
    found_row = ''
    with open(info_file) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[8] == image_name:
                found_row = row
                break

    if found_row == '':
        print('Error -- Image ("{}") not found'.format(image_name))
        sys.exit(-1)

    return found_row[7].split()[0]

def get_label_handinfo_csv(image_name, info_file):
    found_row = ''
    with open(info_file) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[7] == image_name:
                found_row = row
                break

    if found_row == '':
        print('Error -- Image ("{}") not found'.format(image_name))
        sys.exit(-1)

    return found_row[6].split()[0]

def executable(labeled_image_folder,labeled_info_file,unlabeled_image_folder,ground_truth_of_labeling=None,k=30):
    # print labeling results and accuracy
    if ground_truth_of_labeling!=None:
        print('Image:           Classified as:          Actual:')
        print('------------------------------------------------')
        
        data_matrix_dorsal,data_matrix_palmar=get_data_matrix(labeled_image_folder,labeled_info_file)
        dorsal_latent= PCA(n_components=k).fit(data_matrix_dorsal).components_
        #extract_k_latent_semantics(data_matrix_dorsal,k)
        palmar_latent=PCA(n_components=k).fit(data_matrix_palmar).components_
        #extract_k_latent_semantics(data_matrix_palmar,k)
        #print(dorsal_latent)
        #print(palmar_latent)

        err_count=0;
        file_count_all= len(glob.glob1(unlabeled_image_folder,"*.jpg"))
        for img in os.listdir(unlabeled_image_folder):
            if img.endswith('.jpg'):
                img_path=os.path.join(unlabeled_image_folder,img)
                feature_str=','.join(map(str,get_LBP(img_path)))
                feature=[float(x) for x in feature_str.split(',')]
                #print(feature)
                weight_dorsal=0
                weight_palmar=0
                for la in dorsal_latent:
                    weight_dorsal+=cosine_similarity(feature,la)
                for la in palmar_latent:
                    weight_palmar+=cosine_similarity(feature,la)
                #print(weight_dorsal, weight_palmar)
                #sys.exit()
                actucal_label=get_label_handinfo_csv(img,ground_truth_of_labeling)
                if weight_dorsal>weight_palmar:
                    print(img+'         dorsal'+'           '+actucal_label)
                    if actucal_label!='dorsal':
                        err_count+=1
                else:
                    print(img+'         palmar'+'           '+actucal_label)
                    if actucal_label!='palmar':
                        err_count+=1
        print('Classification accuracy: {0}'.format((file_count_all-err_count)/file_count_all))
    else:
        # print labeling results only
        print('Image:           Classified as:')
        print('-------------------------------')
        
        data_matrix_dorsal,data_matrix_palmar=get_data_matrix(labeled_image_folder,labeled_info_file)
        dorsal_latent=extract_k_latent_semantics(data_matrix_dorsal,k)
        palmar_latent=extract_k_latent_semantics(data_matrix_palmar,k)

        for img in os.listdir(unlabeled_image_folder):
            if img.endswith('.jpg'):
                img_path=os.path.join(unlabeled_image_folder,img)
                feature_str=','.join(map(str,get_LBP(img_path)))
                feature=[float(x) for x in feature_str.split(',')]
                weight_dorsal=0
                weight_palmar=0
                for la in dorsal_latent:
                    weight_dorsal+=cosine_similarity(feature,la)
                for la in palmar_latent:
                    weight_palmar+=cosine_similarity(feature,la)

                #print(f'weight_dorsal: {weight_dorsal}')
                #print(f'weight_palmar: {weight_palmar}')
                if weight_dorsal>weight_palmar:
                    print(img+'         dorsal')
                else:
                    print(img+'         palmar')
 