import torch
from torch import nn, optim
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
import random

import numpy as np
import scipy.sparse as sp
import scipy.io as scio

import numpy.matlib
import pickle
import csv
import os

import argparse




from model import test
from utils import dotdict, load_data, normalized_hypergraph



    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='UMMA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--period', type=str, nargs='?', default='delicious1', help="period")
    parser.add_argument('--dataname', type=str, nargs='?', default='delicious', help="period")
    setting = parser.parse_args()
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    device = torch.cuda.current_device()
    period = setting.period
    dataname = setting.dataname
    
    
    X_ori, U, U_baseline, Z, index_hist, index_training, index_test = load_data('data/' + period + '/' + dataname + '.mat', 'data/' + dataname + '_semantic.pkl')
    
    
    X = normalized_hypergraph(X_ori)
    U = np.eye(X.shape[0])

    
    no = Z.sum(1)
    no[no == 0] = 1
    no = np.matlib.repmat(no, Z.shape[1], 1).transpose()
    Z = Z / no

    
    data = dotdict()
    args = dotdict()
    
    
    data.X = X
    data.U = U
    data.Z = Z
    data.X_ori = X_ori
    
    data.index_hist = index_hist
    data.index_training = index_training
    data.index_test = index_test
    
    
    args.epochs = 300
    args.verb = 1
    
    
    
    if dataname == 'delicious':
        hidden_features = 512
        out_features = 256
        learning_rate = 0.0005
        rho = 0.01 
        gamma = 0.01
        dropout = 0
                          

    name = f'{learning_rate}-{dropout}-{hidden_features}-{out_features}-{rho}-{gamma}'
    print(f'Training the config: {name} ...')


    args.learning_rate = learning_rate
    args.dropout = dropout
    args.rho = rho
    args.gamma = gamma
    args.hidden_features = hidden_features
    args.out_features = out_features
    args.k = 20

    test(data, args)
        
        
        
        
        
        

                                   
