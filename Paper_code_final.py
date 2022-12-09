#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:41:27 2022

@author: evanunnink
"""

#packages
import re
import pandas as pd
import numpy as np
from random import randint
import sys
from collections import defaultdict
from itertools import combinations 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
#import collections
#import matplotlib.pyplot as plt
#import scipy.spatial.distance as ssd
#import statistics
from sklearn.cluster import AgglomerativeClustering

#%% Data import and datacleaning
# Import data
import json
f = open("TVs-all-merged.json")
data = json.load(f)
f.close()

#%%
# Data Cleaning
#all possible inch variations. Becomes "inch"
inchlist = [
    '"',
    ' "',
    "inch",
    "inches",
    "-inch",
    "-inches",
    " inch",
    " inches"
    ]

# all possible Hertz variations. Becomes "hz"
hzlist = [
    "hertz",
    "hz",
    "-hz",
    " hz",
    "-hertz"
    ]

# all possible lbs variations. Becomes "lbs"
lbslist = [
    "lbs",
    "lbs.",
    "lbs,"
    "pounds",
    "pound"
    ]

#%%
for key in data:
    for index in range(len(data[key])):
        #change uppercase to lowercase in title
        data[key][index]["title"] = data[key][index]["title"].lower()
        
        #remove webshop name in title
        data[key][index]["title"] = data[key][index]["title"].replace("best buy", "")
        data[key][index]["title"] = data[key][index]["title"].replace("newegg.com", "")
        data[key][index]["title"] = data[key][index]["title"].replace("amazon.com", "")
        data[key][index]["title"] = data[key][index]["title"].replace("thenerds.net", "")
        
        #replace possible inch variations to "inch"
        for inch_variation in inchlist:
            if inch_variation in data[key][index]["title"]:
                data[key][index]["title"] = data[key][index]["title"].replace(inch_variation, "inch")
        #replace possible hz variations to "hz"
        for hz_variation in hzlist:
            if hz_variation in data[key][index]["title"]:
                data[key][index]["title"] = data[key][index]["title"].replace(hz_variation, "hz") 
        #replace possible lbs variations to "lbs"
        for lbs_variation in lbslist:
            if lbs_variation in data[key][index]["title"]:
                data[key][index]["title"] = data[key][index]["title"].replace(lbs_variation, "lbs")
        
        #remove non-alphanumeric values in title
        data[key][index]["title"] = re.sub(r'[^a-zA-Z0-9 ]', '', data[key][index]["title"])
        #remove whitespace at beginning and end op title
        data[key][index]["title"] = data[key][index]["title"].lstrip()
        
        # changes to featuresmap
        for keyF in data[key][index]["featuresMap"]:
            #change uppercase to lowercase in feauturesmap
            data[key][index]["featuresMap"][keyF] = data[key][index]["featuresMap"][keyF].lower()
            #replace possible inch variations to "inch"
            for inch_variation in inchlist:
                if inch_variation in data[key][index]["featuresMap"][keyF]:
                    data[key][index]["featuresMap"][keyF] = data[key][index]["featuresMap"][keyF].replace(inch_variation, "inch")
            #replace possible hz variations to "hz"
            for hz_variation in hzlist:
                if hz_variation in data[key][index]["featuresMap"][keyF]:
                    data[key][index]["featuresMap"][keyF] = data[key][index]["featuresMap"][keyF].replace(hz_variation, "hz") 
            #replace possible lbs variations to "lbs"
            for lbs_variation in lbslist:
                if lbs_variation in data[key][index]["featuresMap"][keyF]:
                    data[key][index]["featuresMap"][keyF] = data[key][index]["featuresMap"][keyF].replace(lbs_variation, "lbs")
            #remove non-alphanumeric values in featuresmap
            data[key][index]["featuresMap"][keyF] = re.sub(r'[^a-zA-Z0-9 ]', '', data[key][index]["featuresMap"][keyF])

#%% Model ID, shops
    model_ID = []
    for key in data:
        for index in range(len(data[key])):
            model = data[key][index]["modelID"]
            model_ID.append(model)
    N = len(model_ID)

    webshop = []
    for key in data:
        for index in range(len(data[key])):
            shop = data[key][index]["shop"]
            webshop.append(shop)
#%% Functions
#%% Function 1
#transfer data from title into a list
def words_titles(title):
    title_sep = []
    for i in range(len(title)):
        title_part = title[i].split()
        title_sep.append(title_part)
    words_in_title = []
    #Create list with all individual words
    for index in title_sep:
        for j in index:
                words_in_title.append(j)
    return words_in_title
       
#%% Funnction 2
#create modelwords from words in title
def model_words_f(words_in_title):
    model_words = []
    for index in range(len(words_in_title)):
        if re.search("([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", words_in_title[index]):
            model_words.append(words_in_title[index])
    model_words =list(set(model_words))
    return model_words
            
#%% Function 3 
#Binary matrix
def binary_matrix_f(title, model_words):
    binary_list = []
    for index in range(len(title)):
        binary_vector = []
        for word in range(len(model_words)):
            if model_words[word] in title[index]:
                binary_vector.append(1)
            else:
                binary_vector.append(0)
        binary_list.append(binary_vector)
    binary_matrix = pd.DataFrame(binary_list).T
    binary_matrix = binary_matrix.to_numpy()
    return binary_matrix


#%%  Function 4
#hash function -> signature matrix
def is_prime(n):
  for i in range(2, n):
    if (n%i) == 0:
      return False
  return True

def compute_p(model_words):
    p = len(model_words)
    while is_prime(p) == False:
        p = p + 1
    return p

#hash function
def hash_function(a, b , x, p):
    hash = (a + b*x) % p
    return hash

def signature_matrix(binary_matrix, model_words):
    k = 720 #highly composite number
    c = binary_matrix.shape[1] # number of columns in signature matrix
    a = [randint(0, len(model_words)) for i in range(k)]
    b = [randint(0, len(model_words)) for i in range(k)]
    
    sig_matrix = np.ones((k , c)) * sys.maxsize
    p = compute_p(model_words)
    for r in range(len(binary_matrix)):
        hash_value_list = []
        for i in range(k):
            hash_value = hash_function(a[i], b[i], r+1, p)
            hash_value_list.append(hash_value)
        for columns in range(c):
            if binary_matrix[r][columns] == 1: 
                for i in range(k):
                  if hash_value_list[i] < sig_matrix[i][columns]:
                      sig_matrix[i][columns] = hash_value_list[i]
    return sig_matrix
            
#%% Function 5
#Number of bands and rows
def b_r(k):
    n = k
    b_values = []
    r_values = []
    
    for b in range(1, (n+1)):
        r = n/b
        if r.is_integer():
            b_values.append(b)
            r_values.append(r)
    br_values = (b_values, r_values)
    br_values = pd.DataFrame(br_values).T
    br_values = br_values.to_numpy()
    # Threshold values
    threshold_values = np.zeros((len(b_values), 3))
    for i in range(len(b_values)):
        threshold = (1/b_values[i])**(1/r_values[i])
        threshold_values[i,0] = b_values[i]
        threshold_values[i,1] = r_values[i]
        threshold_values[i,2] = threshold
    threshold_values = threshold_values[np.logical_and(threshold_values[:, 2] >= 0.4, threshold_values[:, 2]<0.65)]
    # b and r where threshold close to 70
    # index = np.abs(threshold_values[:,2] - 0.70).argmin()
    # b_final = b_values[index]
    # r_final = r_values[index]
    # threshold = (threshold_values, b_final, r_final)
    return threshold_values[:,0]

#%% Function 6
#buckets
def bands_f(sig_matrix, b_final):
    bands = np.array_split(sig_matrix, b_final, axis=0)
    hash_bucket = defaultdict(set)
    
    for band in range(len(bands)):
        for index, product in enumerate(bands[band].transpose()):
            minhash_product = tuple(product)
            hash_bucket[minhash_product].add(index)
    return hash_bucket   

#%% Function 7
#candidate pairs --> matrix
def candidate_matrix_f(hash_bucket, N):
    candidate_pairs = set()
    #total_list = []
    for bucket in hash_bucket:
        if len(hash_bucket[bucket]) >1:
            pairs = list(combinations(list(hash_bucket[bucket]), 2))
            candidate_pairs.update(pairs)
    candidate_matrix = np.zeros((N,N))
    for pair in candidate_pairs:
        candidate_matrix[pair[0], pair[1]] = 1
        candidate_matrix[pair[1], pair[0]] = 1
    return candidate_pairs, candidate_matrix

#%% Function 8
#dissimilarity matrix
def jaccard(A, B):
    nominator = np.logical_and(A, B)
    denominator = np.logical_or(A,B)
    similarity = nominator.sum()/float(denominator.sum())
    dissimilarity = 1 - similarity
    if denominator.sum() == 0:
        dissimilarity = sys.maxsize()
    return dissimilarity

def dissimilarity_matrix_f(candidate_matrix, binary_matrix, webshop, N):
    dissimilarity_matrix = np.empty((N, N))
    for row in range(len(candidate_matrix)):
        for column in range(len(candidate_matrix)):
            if candidate_matrix[row, column] == 0 or webshop[row] == webshop[column]:
                dissimilarity_matrix[row, column] = sys.maxsize
            if candidate_matrix[row, column] == 1 and webshop[row] != webshop[column]:
                dissimilarity_matrix[row, column] = jaccard(binary_matrix[:, row], binary_matrix[:, column]) 
            dissimilarity_matrix[row,row]=0
    return dissimilarity_matrix

#dissimilarity_matrix_check = pd.DataFrame(dissimilarity_matrix, columns= model_ID, index= model_ID)
                
#%% Function 9
#hierarchical clustering - completelinkage

def clustering(dissimilarity_matrix, model_ID, d):
    cluster = AgglomerativeClustering(n_clusters=None, affinity='precomputed', memory=None, connectivity=None, compute_full_tree=True, linkage='complete', distance_threshold=d, compute_distances=False).fit(dissimilarity_matrix)
    labels = cluster.labels_
    
    clusters = dict()
    for cluster_nr in range(0, max(labels)):
        indices = list(np.where(labels == cluster_nr))
        clusters[cluster_nr] = list([model_ID[i] for i in indices[0]])       
    return clusters
        
#%% Functino 10
#evaluation LSH

    #pair quality = number of duplicates found/number of comparisons made
def evaluationLSH(candidate_pairs, model_ID):
    duplicates_found = 0
    for i in range(len(candidate_pairs)):
        if model_ID[list(candidate_pairs)[i][0]] == model_ID[list(candidate_pairs)[i][1]]:
            duplicates_found +=1
    pair_quality = duplicates_found/len(candidate_pairs)
    #pair completeness = number of duplicates found/total number of duplicates 
    total_duplicates = 0
    for i in range(len(model_ID)):
        for j in range(i+1, len(model_ID)):
            if model_ID[i] == model_ID[j]:
                total_duplicates += 1
    pair_completeness = duplicates_found/total_duplicates
    f_1_LSH = (2*pair_quality*pair_completeness)/(pair_quality + pair_completeness)
    
    possible_comparisons = (len(model_ID)*(len(model_ID)-1))/2
    fraction_comparion = len(candidate_pairs)/possible_comparisons
    return pair_quality, pair_completeness, f_1_LSH, fraction_comparion

#%% Function 11
#evaluation clustering 
def evaluationHC(clusters, model_ID):
    cluster_pairs = set()
    for i in clusters: 
        pairs = list(combinations(clusters[i], 2))
        cluster_pairs.update(pairs)
        
   # precision = number of duplicates found/number of comparisons made
    duplicates_found_HC = 0
    for i in range(len(cluster_pairs)):
        if list(cluster_pairs)[i][0] == list(cluster_pairs)[i][1]:
            duplicates_found_HC +=1
    precision = duplicates_found_HC/len(cluster_pairs)
    #recall = number of duplicates found/total number of duplicates 
    total_duplicates_HC = 0
    for i in range(len(model_ID)):
        for j in range(i+1, len(model_ID)):
            if model_ID[i] == model_ID[j]:
                total_duplicates_HC += 1
    recall = duplicates_found_HC/total_duplicates_HC
    f_1_HC = (2*precision*recall)/(precision+recall)
    return   precision, recall, f_1_HC, cluster_pairs

#%% Function 12
#bootstrapping
def bootstrapping(title, model_ID, webshop):
    bootstrap =[randint(0, 1623) for i in range(1624)]
    training_set = np.unique(bootstrap)
    
    test_set = []
    for x in range(1624):
        if x not in training_set:
            test_set.append(x)
    
    titles_train = list(title[i] for i in training_set)
    titles_test = list(title[i] for i in test_set)
    
    N_train = len(titles_train)
    N_test = len(titles_test)
    model_ID_train = [model_ID[index] for index in training_set]
    model_ID_test = [model_ID[index] for index in test_set]   
    
    webshop_train = [webshop[index] for index in training_set]  
    webshop_test = [webshop[index] for index in test_set]  
    return titles_train, titles_test, N_train, N_test, model_ID_train, model_ID_test, webshop_train, webshop_test

#%% Main 1
N = 1624
title = []
for key in data:
   for index in data[key]:
       title_list = index['title']
       title.append(title_list)
b_values = b_r(720)
d_thres = np.arange(0.3, 0.6, 0.1)
#%% Main 2
#training --> determine distance threshold

PQ_mean = []
PC_mean = []
f_1_lsh_mean = []
precision_mean = []
recall_mean = []
f_1_mean = []

for b in b_values:
    for d in d_thres:
        print(d)
        PQ = []
        PC = []
        FC = []
        f_1_lsh = []
        f_1 = []
        prec = []
        rec = []
        for strap in range(5):
            BS = bootstrapping(title, model_ID, webshop)
            titles_train = BS[0]
            N_train = BS[2]
            MID_train = BS[4]
            WS_train = BS[6]
            WT_train = words_titles(titles_train)
            MW_train = model_words_f(WT_train)
            BM_train = binary_matrix_f(titles_train, MW_train)
            SM_train = signature_matrix(BM_train, MW_train)
            bands_train = bands_f(SM_train, b)
            C_train = candidate_matrix_f(bands_train, N_train)
            CP_train = C_train[0]
            CM_train = C_train[1]
            DM_train = dissimilarity_matrix_f(CM_train, BM_train, WS_train, N_train)
            cluster_train = clustering(DM_train, MID_train, d)
            eval_LSH_train = evaluationLSH(CP_train, MID_train)
            PQ_train = eval_LSH_train[0]
            PC_train = eval_LSH_train[1]
            f_1_LSH_train = eval_LSH_train[2]
            FC_train = eval_LSH_train[3]
            eval_HC_train = evaluationHC(cluster_train, MID_train)
            prec_train = eval_HC_train[0]
            recall_train = eval_HC_train[1]
            f_1_HC_train = eval_HC_train[2]
            cluster_pairs_train = eval_HC_train[3]
            #create list with statistics values for a b
            PQ.append(PQ_train)
            PC.append(PC_train)
            f_1_lsh.append(f_1_LSH_train)
            FC.append(FC_train)
            prec.append(prec_train)
            rec.append(recall_train)
            f_1.append(f_1_HC_train)
        #create average of stats for a b
        PQ_mean.append(np.mean(PQ))
        PC_mean.append(np.mean(PC))
        f_1_lsh_mean.append(np.mean(f_1_lsh))
        precision_mean.append(np.mean(prec))
        recall_mean.append(np.mean(rec))
        f_1_mean.append(np.mean(f_1))
        
#%% Main 3
#distance threshold = 0.5
PQ_mean = []
PC_mean = []
f_1_lsh_mean = []
f_1_mean = []
FC_mean = []
precision_mean = []
recall_mean = []

for b in range(1, 721, 100):
    print(b)
    PQ = []
    PC = []
    FC = []
    f_1_lsh = []
    FC = []
    f_1 = []
    for strap in range(5):
        print(strap)
        BS = bootstrapping(title, model_ID, webshop)
        titles_train = BS[0]
        N_train = BS[2]
        MID_train = BS[4]
        WS_train = BS[6]
        WT_train = words_titles(titles_train)
        MW_train = model_words_f(WT_train)
        BM_train = binary_matrix_f(titles_train, MW_train)
        SM_train = signature_matrix(BM_train, MW_train)
        bands_train = bands_f(SM_train, b)
        C_train = candidate_matrix_f(bands_train, N_train)
        CP_train = C_train[0]
        CM_train = C_train[1]
        print("test")
        DM_train = dissimilarity_matrix_f(CM_train, BM_train, WS_train, N_train)
        cluster_train = clustering(DM_train, MID_train, 0.5)
        eval_LSH_train = evaluationLSH(CP_train, MID_train)
        PQ_train = eval_LSH_train[0]
        PC_train = eval_LSH_train[1]
        f_1_LSH_train = eval_LSH_train[2]
        FC_train = eval_LSH_train[3]
        eval_HC_train = evaluationHC(cluster_train, MID_train)
        prec_train = eval_HC_train[0]
        recall_train = eval_HC_train[1]
        f_1_HC_train = eval_HC_train[2]
        PQ.append(PQ_train)
        PC.append(PC_train)
        f_1_lsh.append(f_1_LSH_train)
        FC.append(FC_train)
        f_1.append(f_1_HC_train)
    #values for every b
    PQ_mean.append(np.mean(PQ))
    PC_mean.append(np.mean(PC))
    f_1_lsh_mean.append(np.mean(f_1_lsh))
    f_1_mean.append(np.mean(f_1))
    precision_mean.append(np.mean(prec_train))
    recall_mean.append(np.mean(recall_train))
    FC_mean.append(FC)


#%% graphs 
plt.plot(FC_mean, PC_mean)
plt.xlabel("Fraction of comparison")
plt.ylabel("Pair completeness")
plt.plot(FC_mean, PQ_mean)
plt.xlabel("Fraction of comparison")
plt.ylabel("Pair quality")
plt.plot(FC_mean, f_1_mean)
plt.xlabel("Fraction of comparison")
plt.ylabel("F_1 measure")


