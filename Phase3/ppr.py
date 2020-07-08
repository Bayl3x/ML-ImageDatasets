#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import random

import utils

class PPR:
    def __init__(self, k, image_names):
        self.k = k
        self.image_names = image_names
        self.test_names = []
    """        
    def get_image_names_with_label(self, fileName, feature_model, label):
        df = pd.read_csv(fileName, header=None)
        image_names = []
        for row in df.values:
            if row[2] == label:
                image_names.append(row[0])
        return image_names
    """
    
    def compute_and_store_ii_similarity(self, dm, inames, output_file):
        dmdict = {}
        length = len(inames)
        #calculate cosine similarity for all of images for all images
        for i in range(length):
            dmdict[inames[i]] = utils.cosine_similarity_mv(dm, dm[i])
            #print("{} remaining".format(length - i))
        #create a data frame with correct column and row names corresponding to the images
        df = pd.DataFrame.from_dict(dmdict, orient='index')
        df.columns = inames
    
        #write to csv
        df.to_csv(output_file)

    #get top k similarty graph for each images.
    #uses the ii_similarity matrix, sort by distance, get top K, and prune the rest
    # also getting and returning the adjacency matrix here since the same values 
    #  for the graph can be used to generate the matrix
    def get_k_similartiy_graph(self, similarityMatrix, k):
        
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
    def ppr(self, graph, image_names, adjacencyMatrix, image1, image2=None, image3=None):
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

    def fit(self, x_train, y_train):
        x_train = x_train.astype(int)
        self.x_train = x_train
        self.y_train = y_train
        inames = self.image_names
        self.compute_and_store_ii_similarity(x_train, inames, '../proj3t4.csv')

    def predict(self, x_test):
        x_test = x_test.astype(int)
        y_pred = []
        for i in range(len(x_test)):
            temp_array = np.vstack([self.x_train, x_test[i]])
            temp_names = np.append(self.image_names, self.test_names[i])
            self.compute_and_store_ii_similarity(temp_array, temp_names, '../proj3t4.csv')
            df = pd.read_csv('../proj3t4.csv', index_col=0)

            graph, adjacencyMatrix = self.get_k_similartiy_graph(df, self.k)
            state_prob_vector = self.ppr(graph, temp_names, adjacencyMatrix, temp_names[-1])
            dorsal_sum = 0
            palmar_sum = 0
            for j in range(len(state_prob_vector) - 1):
                if self.y_train[j] == 'dorsal':
                    dorsal_sum = dorsal_sum + state_prob_vector[j]
                elif self.y_train[j] == 'palmar':
                    palmar_sum = palmar_sum + state_prob_vector[j]
            if dorsal_sum > palmar_sum:
                y_pred.append('dorsal')
            elif palmar_sum > dorsal_sum:
                y_pred.append('palmar')
            # Tiebreaker -- need to figure out
            else:
                dorsal_count = 0
                palmar_count = 0
                image_index = np.argsort(state_prob_vector)[::-1][1:(self.k + 1)]
                for ind in image_index:
                    if self.y_train[ind] == 'dorsal':
                        dorsal_count = dorsal_count + 1
                    elif self.y_train[ind] == 'palmar':
                        palmar_count = palmar_count + 1
                
                if dorsal_count > palmar_count:
                    y_pred.append('dorsal')
                else:
                    y_pred.append('palmar')
                    
                """
                rand = random.randint(0, 1)
                if rand == 0:
                    y_pred.append('dorsal')
                else:
                    y_pred.append('palmar')
                """
        return y_pred

    def predict_t6(self, x_test):
        x_test = x_test.astype(int)
        y_pred = []
        counter = 0
        percentage = 0
        print("Processing...")
        for i in range(len(x_test)):
            counter = counter + 1
            if counter % 110 == 0:
                percentage = percentage + 1
                print(percentage, "%",)
            if self.test_names[i] in self.image_names:
                img_index = self.image_names.index(self.test_names[i])
                y_pred.append(self.y_train[img_index])
            if self.test_names[i] not in self.image_names:
                temp_array = np.vstack([self.x_train, x_test[i]])
                temp_names = np.append(self.image_names, self.test_names[i])
                self.compute_and_store_ii_similarity(temp_array, temp_names, '../proj3t4.csv')
                df = pd.read_csv('../proj3t4.csv', index_col=0)
    
                graph, adjacencyMatrix = self.get_k_similartiy_graph(df, self.k)
                state_prob_vector = self.ppr(graph, temp_names, adjacencyMatrix, temp_names[-1])

                relevant_sum = 0
                irrelevant_sum = 0
                for j in range(len(state_prob_vector) - 1):
                    if self.y_train[j] == 0:
                        irrelevant_sum = irrelevant_sum + state_prob_vector[j]
                    elif self.y_train[j] == 1:
                        relevant_sum = relevant_sum + state_prob_vector[j]
                if irrelevant_sum > relevant_sum:
                    y_pred.append(0)
                elif relevant_sum > irrelevant_sum:
                    y_pred.append(1)
                # Tiebreaker
                else:
                    irrelevant_count = 0
                    relevant_count = 0
                    image_index = np.argsort(state_prob_vector)[::-1][0:(self.k+1)]
                    for ind in image_index:
                        if ind < len(self.y_train):
                            if self.y_train[ind] == 0:
                                irrelevant_count = irrelevant_count + 1
                            elif self.y_train[ind] == 1:
                                relevant_count = relevant_count + 1
                    
                    if irrelevant_count > relevant_count:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
        return y_pred             
        """
        df = pd.read_csv(self.output_file, index_col=0)
        # Using 5 for now; will want to test more to see which values of k work best
        graph, adjacencyMatrix = self.get_k_similartiy_graph(df, self.k)

        image_names = df.columns
        dorsal_images = self.get_image_names_with_label('../phase3.csv', 'LBP', 'dorsal')
        palmar_images = self.get_image_names_with_label('../phase3.csv', 'LBP', 'palmar')
        dorsal_sum = 0
        palmar_sum = 0
        state_prob_vector = self.single_ppr(graph, image_names, adjacencyMatrix, image_id)
        for i in range(len(state_prob_vector)):
            image_name = image_names[i]
            if image_name in dorsal_images:
                dorsal_sum = dorsal_sum + state_prob_vector[i]
            elif image_name in palmar_images:
                palmar_sum = palmar_sum + state_prob_vector[i]

        if dorsal_sum > palmar_sum:
            print('dorsal')
        elif palmar_sum > dorsal_sum:
            print('palmar')
        """