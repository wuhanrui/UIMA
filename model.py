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

from utils import *

def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
               
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input_features, adj):        
        support = torch.mm(input_features, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class UIMA(nn.Module):
    def __init__(self, in_featuresX, in_featuresY, hidden_features, out_features, dropout):
        super(UIMA, self).__init__()
        
        self.dropout = dropout

        # encoder X
        self.encoderX_0 = GraphConvolution(in_featuresX, hidden_features)
        self.encoderX_1 = GraphConvolution(hidden_features, out_features)
        
        # encoder Y
        self.encoderY_0 = nn.Linear(in_featuresY, hidden_features)
        self.encoderY_1 = nn.Linear(hidden_features, out_features)
        
        
        # decoder Y
        self.decoderY_0 = nn.Linear(out_features, hidden_features)
        self.decoderY_1 = nn.Linear(hidden_features, in_featuresY)
        

    def forward(self, x, hx, y, index_hist, index_training, index_test):
        
        x = self.encoderX_0(x, hx)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout)
 
        x = self.encoderX_1(x, hx)
        x = torch.tanh(x)
        
        
        y = self.encoderY_0(y)
        y = torch.tanh(y)  
        y = F.dropout(y, self.dropout)
        
        y = self.encoderY_1(y)
        y = torch.tanh(y)   
        
        
        index_hist = index_hist.cpu()
        index_training = index_training.cpu()
        index_test = index_test.cpu()
        y = y.cpu()
        
        index_hist = torch.LongTensor(index_hist)
        index_training = torch.LongTensor(index_training)
        index_test = torch.LongTensor(index_test)
        y_hist = torch.index_select(y, 0, index_hist)
        y_test = torch.index_select(y, 0, index_test)
        y_training = torch.index_select(y, 0, index_training)
        
        
        

        y = y.cuda()
        y_hist = y_hist.cuda()
        y_training = y_training.cuda()
        y_test = y_test.cuda()
        index_hist = index_hist.cuda()
        index_training = index_training.cuda()
        index_test = index_test.cuda()
        
        
        
        # decode hypergraph
        h = torch.mm(x, y_hist.t())
        h = torch.sigmoid(h)
        
        
        # decode semantic
        y = self.decoderY_0(y)
        y = torch.tanh(y)
        y = F.dropout(y, self.dropout)
        
        
        y = self.decoderY_1(y)
        y = torch.tanh(y)
        
        return h, x, y, y_training, y_test

      
        
def test(data, args, s = 616):

    seed_everything(seed = s)


    X = data.X
    X_ori = data.X_ori
    U = data.U
    Z = data.Z


    index_hist = data.index_hist
    index_training = data.index_training
    index_test = data.index_test
    test_list = np.unique(index_test)


    epochs = args.epochs
    dropout = args.dropout
    learning_rate = args.learning_rate
    rho = args.rho
    gamma = args.gamma
    hidden_features = args.hidden_features
    out_features = args.out_features
    k = args.k


    in_featuresU = U.shape[1]
    in_featuresZ = Z.shape[1]



    X = torch.from_numpy(X).float()
    U = torch.from_numpy(U).float()
    Z = torch.from_numpy(Z).float()
    index_hist = torch.from_numpy(index_hist).long()
    index_training = torch.from_numpy(index_training).long()
    test_list = torch.from_numpy(test_list).long()


    model = UIMA(in_featuresU, in_featuresZ, hidden_features, out_features, dropout)
    model = nn.DataParallel(model)
    model.cuda()

    criteon = nn.MSELoss()
    criteon2 = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):

        U = U.cuda()
        X = X.cuda()
        Z = Z.cuda()
        index_hist = index_hist.cuda()
        index_training = index_training.cuda()
        test_list = test_list.cuda()

        tilde_X, tilde_U, tilde_Z, Z_training, Z_test = model(U, X, Z, index_hist, index_training, test_list)

        tilde_X = tilde_X.cpu()
        tilde_X = torch.FloatTensor(tilde_X)
        X_ori = torch.FloatTensor(X_ori)
        tilde_X = tilde_X.cuda()
        X_ori = X_ori.cuda()

        loss1 = rho * criteon2(tilde_X, X_ori) 
        loss2 = gamma * criteon(tilde_Z, Z) 
        loss3 = criteon(tilde_U, Z_training)
        loss  = loss1 + loss2 + loss3
        X_ori = X_ori.cpu()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    with torch.no_grad():

        tilde_U = tilde_U.cpu()
        Z_test = Z_test.cpu()
        Z_test = Z_test.numpy()
        item_test_list = test_list.cpu().numpy()

        true_interacted_movies_list = []
        recommend_movie_id_list = []
        for i in range(tilde_U.shape[0]):

            user = tilde_U[i]
            rep_user = np.matlib.repmat(user, Z_test.shape[0], 1)

            score = np.sum(np.power(rep_user - Z_test, 2), 1)
            ranking = np.argsort(score)

            # Predict
            prediction = item_test_list[ranking].tolist()
            recommend_movie_id_list.append(prediction)

            # Real
            real = [index_test[i].tolist()]
            true_interacted_movies_list.append(real)


        p20 = precision_at_k(true_interacted_movies_list, recommend_movie_id_list, args.k)
        ndcg20 = ndcg_k(true_interacted_movies_list, recommend_movie_id_list, args.k)
        hr20 = hitrate_at_k(true_interacted_movies_list, recommend_movie_id_list, args.k)
        print("Precision@20: ", '%.4f|' % p20, "NDCG@20: ", '%.4f|' % ndcg20, "HitRate@20: ", '%.4f|' % hr20)


                              
            
