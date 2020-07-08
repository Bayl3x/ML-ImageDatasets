#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:22:17 2019

@author: hangzhao
"""

import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn import preprocessing

#Kernel function
def linear(x, y):
    return np.dot(x, y)

def polynomial(x, y, p):
    return (1 + np.dot(x, y)) ** p

def rbf(x, y, sigma):
    return np.exp(-np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2)))
           
#SVM
class SVM:
    def __init__(self, kernel='poly', kernel_param = 3, soft_margin = None):
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.soft_margin = soft_margin
        
    #Train
    def fit(self, X, y):
        self.labelEncoder = preprocessing.LabelEncoder()
        y_encoded = self.labelEncoder.fit_transform(y)
        y_encoded[y_encoded == 0] = -1
        n_samples, n_features = X.shape
        
        #Apply kernel function (gram matrix)
        Kernel = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'linear':
                    Kernel[i,j] = linear(X[i], X[j])
                if self.kernel == 'poly':
                    Kernel[i,j] = polynomial(X[i], X[j], self.kernel_param)
                if self.kernel == 'rbf':
                    Kernel[i,j] = rbf(X[i], X[j], self.kernel_param)
        
        # Get Lagrange multipliers
        P = cvxopt.matrix(np.outer(y_encoded,y_encoded) * Kernel)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y_encoded, (1, n_samples))
        A = cvxopt.matrix(A, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)
        
        if self.soft_margin is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.soft_margin)))
            
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        #Lagrange multipliers
        alpha = np.ravel(solution['x'])

        #Get support vectors
        sv = alpha > 0
        index = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y_encoded[sv]
        print("%d support vectors out of %d points" % (len(self.alpha), n_samples))

        #Calculate b
        self.b = 0
        for n in range(len(self.alpha)):
            self.b = self.b + self.sv_y[n]
            self.b = self.b - np.sum(self.alpha * self.sv_y * Kernel[index[n], sv])
        self.b = self.b / len(self.alpha)

        #Calculate w for linear Kernel
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.alpha)):
                self.w = self.w + self.alpha[n] * self.sv_y[n] * self.sv[n]
            
    #Predict
    def predict(self, X):
        if self.kernel == 'linear':
            y_predict = np.sign(np.dot(X, self.w) - self.b)
            y_pred_encoded = np.array(y_predict).astype(int)
            y_pred_encoded[y_pred_encoded == -1] = 0
            prediction = self.labelEncoder.inverse_transform(y_pred_encoded)
            
            return prediction
        
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                    if self.kernel == 'poly':
                        s = s + (a * sv_y * polynomial(X[i], sv, self.kernel_param))
                    if self.kernel == 'rbf':
                        s = s + (a * sv_y * rbf(X[i], sv, self.kernel_param))
                y_predict[i] = s
            
            y_pred_encoded = np.array(np.sign(y_predict - self.b)).astype(int)
            y_pred_encoded[y_pred_encoded == -1] = 0
            prediction = self.labelEncoder.inverse_transform(y_pred_encoded)
            
            return prediction
