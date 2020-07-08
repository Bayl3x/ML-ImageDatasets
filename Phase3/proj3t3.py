import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils, utils3



parser = argparse.ArgumentParser(description='Creates image-image similarity graph with K edges. Then uses PPR to find most dominant images.')
parser.add_argument('csv_file', metavar='CSVFILE', help='csv file where image-image similarity matrix is stored')
parser.add_argument('k', metavar='K', help='Number of edges for the graph to have')
parser.add_argument('K', metavar='KIMAGES', help='Number of most dominant images to return')
parser.add_argument('image_id1', metavar='IMAGEID1', help='ID of first image to use for PPR')
parser.add_argument('image_id2', metavar='IMAGEID2', help='ID of second image to use for PPR')
parser.add_argument('image_id3', metavar='IMAGEID3', help='ID of third image to use for PPR')
parser.add_argument('directory', metavar='DIRECTORY', help='Folder containing the images to be used so that they can be displayed')

args = parser.parse_args()

df = pd.read_csv(args.csv_file, index_col=0)

graph, adjacencyMatrix = utils3.get_k_similartiy_graph(df,int(args.k))
"""
state_prob_vector1 = utils3.ppr(graph, df.columns, adjacencyMatrix, args.image_id1)
state_prob_vector2 = utils3.ppr(graph, df.columns, adjacencyMatrix, args.image_id2)
state_prob_vector3 = utils3.ppr(graph, df.columns, adjacencyMatrix, args.image_id3)

avg_state_prob_vector = []
for i in range(len(state_prob_vector1)):
	summation = state_prob_vector1[i] + state_prob_vector2[i] + state_prob_vector3[i]
	avg = summation / 3
	avg_state_prob_vector.append(avg)
"""

state_prob_vector = utils3.ppr(graph, df.columns, adjacencyMatrix, args.image_id1, args.image_id2, args.image_id3)
image_index = np.argsort(state_prob_vector)[::-1][0:int(args.K)]
print(df.index.values[image_index])

for i in image_index:
    plt.imshow(utils.load_image_rgb(os.path.join(args.directory, df.index.values[i])))
    plt.title(df.index.values[i])
    plt.show()


""" DELETE ENTIRE CHUNK ONCE WERE SATISFIED, JUST HERE FOR DISECTING
i1 = state_prob_vector1.argmax()
i2 = state_prob_vector2.argmax()
i3 = state_prob_vector3.argmax()
print(df.index.values[i1], df.index.values[i2], df.index.values[i3])

Is = np.where(state_prob_vector1 > 0.02)
print(state_prob_vector1[Is])
print(df.index.values[Is])
state_prob_vector1 = np.sort(state_prob_vector1)
print(state_prob_vector1[-10:-1])

"""
