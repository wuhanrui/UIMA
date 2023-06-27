import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as scio
import math


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def normalized_hypergraph(H):
    columnsum = H.sum(0)
    rowsum = H.sum(1)
    
    diagD = 1 / np.sqrt(rowsum)
    diagB = 1 / columnsum

    D = np.diag(diagD)
    B = np.diag(diagB)
    Omega = np.eye(H.shape[1])
    
    hx = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)

    return hx 
    
def load_data(filename, semantic_name):

    data_mat = scio.loadmat(filename)
    training_user_data = data_mat['hypergraph']
    hist_movie_list = data_mat['hist_item_list']
    training_movie_list_unique = data_mat['training_item_list_unique']
    test_movie_list_unique = data_mat['test_item_list_unique']
    training_labels = data_mat['training_label']
    test_labels = data_mat['test_label']


    movie_list = np.vstack((hist_movie_list, training_movie_list_unique))
    movie_list = np.vstack((movie_list, test_movie_list_unique))


    with open(semantic_name, 'rb') as f:
        semantic = pkl.load(f)


    Z = np.zeros((len(movie_list), semantic[1].shape[1]))
    for i in range(len(movie_list)):
        Z[i, :] = semantic[movie_list[i].item()]



    update_training_labels = np.zeros((training_labels.shape[0]))
    for i in range(training_labels.shape[0]):
        idx = np.where(training_movie_list_unique == training_labels[i])[0]
        update_training_labels[i] = idx


    index_hist = np.arange(0, len(hist_movie_list), 1)
    index_training = update_training_labels + len(hist_movie_list) 
    index_training = index_training.astype(np.int)




    update_test_labels = np.zeros((test_labels.shape[0]))
    for i in range(test_labels.shape[0]):
        idx = np.where(test_movie_list_unique == test_labels[i])[0]
        update_test_labels[i] = idx


    index_test  = update_test_labels + len(hist_movie_list) + len(training_movie_list_unique) 
    index_test  = index_test.astype(np.int)

    
    
    X = training_user_data
    U = data_mat['feature']
    U_baseline = data_mat['feature_baseline']
    
    return X, U, U_baseline, Z, index_hist, index_training, index_test


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))
    
def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users    
    
def hitrate_at_k(actual, predicted, topk):
    num_hit = 0
    num_interacted = 0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        num_hit += len(act_set & pred_set)
        num_interacted += len(act_set)
        
    return num_hit/num_interacted    
    
    
    
    
    
    
        
        
        