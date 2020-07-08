import csv
import numpy as np
import pandas as pd
import random
import sys
import time
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import hsv_to_rgb
from cycler import cycler
import os
import utils

# import hello

def read_image_fvector(filename, type):
    dataMatrix = []
    test = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        if type == "test":
            for row in reader:
                dataMatrix.append(np.float32(np.asarray(row[6:])))
        else:
            for row in reader:
                if row[2] == type : #read dorsl or pamlar as the user want
                # test.append(np.float32(np.asarray(row[6:])))
                    dataMatrix.append(np.float32(np.asarray(row[6:])))
    # print(dataMatrix)
    return dataMatrix

def read_res(filename):
    res=[]
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            res.append(row[2])
    return res

class KMeansClusterer:
    def __init__(self,ndarray,cluster_num,type):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points=self.__pick_start_point(ndarray,cluster_num)
        self.type = type;

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index=-1
            for i in range(len(self.points)):
                distance = self.__distance(item,self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center=[]
        for item in result:
            new_center.append(self.__center(item).tolist())
        if (self.points==new_center).all():
            return result
        self.points=np.array(new_center)
        return self.cluster()

    def __center(self,list):
        return np.array(list).mean(axis=0)

    def __distance(self,p1,p2):
        tmp=0
        for i in range(len(p1)):
            tmp += pow(p1[i]-p2[i],2)
        return pow(tmp,0.5)

    def __pick_start_point(self,ndarray,cluster_num):
        if cluster_num < 0 or cluster_num > len(ndarray):
        #ndarray.shape[0]:
            raise Exception("Error number.")
        indexes=random.sample(np.arange(0,len(ndarray),step=1).tolist(),cluster_num)
        points=[]
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

    def find_centroid(self, result_listOfCluster):
        # for i in range(len(result_listOfCluster)):
        #     centroid_result.append([])
        centroid_result=[]
        from statistics import mean
        for cluster in result_listOfCluster:
            centroid_result.append(list(map(mean, zip(*cluster))))
        return centroid_result

    def avg_distToOneTypeOfCluster(self, result_listOfCluster, queryimg):
        # queryimg is feature vector
        centroid_result = self.find_centroid(result_listOfCluster)
        total = 0
        for centroid in centroid_result:
            total += self.__distance(centroid,queryimg)
        avg = total / len(centroid_result)
        return avg

    #def cmp_dist()
def dorsalORpalmar(trainigfile, test_filename, p_cluster_num, d_cluster_num):
    test_matrix = read_image_fvector(test_filename, "test")

    p_matrix = read_image_fvector(trainigfile, "palmar")
    d_matrix = read_image_fvector(trainigfile, "dorsal")

    p = KMeansClusterer(p_matrix, p_cluster_num,"palmar")
    p_clusterRes = p.cluster()
    d = KMeansClusterer(d_matrix, d_cluster_num,"dorsal")
    d_clusterRes = d.cluster()

    #colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1]) for i in range(1000)]
    #plt.rc('axes', prop_cycle=(cycler('color', colors)))
    displaypx = []
    displaypy = []
    
    pclusterColors = ['#000099', '#0000CC', '#3333FF', '#0066CC', '#3399FF', '#00FFFF', '#66B2FF']
    dclusterColors = ['#990000', '#CC0000', '#FF0000', '#FF3333', '#FF6666', '#CC0066', '#FF3399']
    i = 0
    for cluster in p_clusterRes:
        x_val = []
        y_val = []
        for c in cluster:
            x_val.append(c[1332])
            y_val.append(c[1908])

        plt.scatter([x_val], [y_val], color = pclusterColors[i % len(pclusterColors)])
        i += 1

        # displaypx.append(x_val)
        # displaypy.append(y_val)
        #plt.scatter(x_val,y_val,color="red")
    # plt.scatter(displaypx,displaypy,color="red",label='palmar')

    displaydx = []
    displaydy = []
    i = 0
    for cluster in d_clusterRes:
        x_val = []
        y_val = []
        for c in cluster:
            x_val.append(c[1332])
            y_val.append(c[1908])

        plt.scatter([x_val], [y_val], color = dclusterColors[i % len(dclusterColors)])
        i += 1

        # displaydx.append(x_val)
        # displaydy.append(y_val)
        #plt.scatter(x_val,y_val,color='blue')
    # plt.scatter(displaydx,displaydy,color="blue",label='dorsal')
    #plt.legend(loc="bottom right")
    plt.savefig('Cluster.png')
    plt.close()

    res = []
    for item in test_matrix:
        # print(item)
        p_avg = p.avg_distToOneTypeOfCluster(p_clusterRes,item)
        d_avg = d.avg_distToOneTypeOfCluster(d_clusterRes,item)
        if p_avg < d_avg:
            res.append(p.type)
        else:
            res.append(d.type)
    # print(res)
    return res

def main(train_image_folder, test_image_folder, info_file, palmar, dorsal):

    labelledCSV="label.csv"
    unlabelledCSV="unlabel.csv"
    try:
        os.remove(labelledCSV)
    except OSError:
        pass
    try:
        os.remove(unlabelledCSV)
    except OSError:
        pass


    label="python proj1t2.py "+train_image_folder+" LBP"+" "+info_file+" "+labelledCSV
    unlabel="python proj1t2.py "+test_image_folder+" LBP"+" "+info_file+" "+unlabelledCSV
    # s="python hello.py 8"
    # print(label)

    labelledCSV="label.csv"
    unlabelledCSV="unlabel.csv"
    label="python proj1t2.py "+train_image_folder+" LBP"+" "+info_file+" "+labelledCSV
    unlabel="python proj1t2.py "+test_image_folder+" LBP"+" "+info_file+" "+unlabelledCSV
    s="python hello.py 8"
    # print(s)
    os.system(label)
    # time.sleep( 5 )
    os.system(unlabel)
    cal = dorsalORpalmar(labelledCSV,unlabelledCSV,palmar, dorsal) # test result
    res = read_res(unlabelledCSV) # actual result

    image_names = utils.get_image_names(unlabelledCSV, 'LBP')

    i=0
    count = 0
    for item in cal:
        if res[i] == item:
            count+=1
        i+=1

    print('Image\t\tClassification\t\tCorrect Label')

    for i in range(len(image_names)):
        print(f'{image_names[i]}\t\t{cal[i]}\t\t{res[i]}')
    print()
    print('Accuracy：',count/len(cal))
    print()
    while True:
        # display 10 images

        i = 0
        while i < 10 and i < len(image_names):
            plt.imshow(utils.load_image_rgb(os.path.join(test_image_folder, image_names[i])))
            plt.title(f'{image_names[i]} -- {cal[i]}')
            plt.show()
            i += 1
        more = input("Display 10 more images? (y/n) >> ")

        if more == 'n' or i >= len(image_names):
            break

    # sample = open('result.txt', 'a')
    # print('accuracy：',count/len(cal), file = sample)
    # sample.close()
