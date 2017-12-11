#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt


def load_train_test_data(train_ratio=.5):
    df = pandas.read_csv('./HTRU_2.csv',header=None)
    data = df.values
    X = numpy.delete(data,8,1)
    X = numpy.concatenate((numpy.ones((len(X),1)),X), axis=1)
    y = data[:,-1].reshape(-1,1)
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def stochastic_gradient_descent(X, y, alpha = .001, iters =1000 , eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    y_hat = numpy.zeros((n,1))
    lamda = 0.001
    tmp = numpy.zeros((d,1))
    for i in range (0,iters):    
        for ins in range (0,n):
            e = -(numpy.dot(X[ins],theta))
            y_hat = 1/(1+numpy.exp(e))
            tmp = (y[ins] - y_hat)
            g = numpy.dot(X[ins].T.reshape(d,1),tmp.reshape(1,1))-lamda
            theta = theta + alpha*g
                     
    return theta

def predict(X, theta):
    return 1/(1+numpy.exp(-(numpy.dot(X,theta))))

def plot_roc_curve(y_hat,y_test):
    n,d = y_hat.shape
    y_p=numpy.hstack((y_hat,y_test))
    y_pd=pandas.DataFrame(data=y_p,columns = ['0','1'])
    y_pds = y_pd.sort_values(by=['0'])
    y_pst = y_pds.values
    r = numpy.zeros((n,1))
    x = numpy.zeros((n,1))
    y = numpy.zeros((n,1))

    for i in range(0,len(y_pst)):
        tp,tn,fn,fp = 0,0,0,0
        for j in range (0,len(y_pst)):
            if (y_pst[j,0]>y_pst[i,0]):
                r[j]=1
            else:
                r[j]=0
            if(r[j]==1 and y_pst[j,1]==1):
                tp = tp+1
            elif(r[j]==0 and y_pst[j,1]==0):
                tn = tn+1
            elif(r[j]==1 and y_pst[j,1]==0):
                fp = fp + 1
            elif(r[j]==0 and y_pst[j,1]==1):
                fn = fn + 1
    
        y[i]= tp/(tp+fn)
        x[i] = fp/(fp+tn)
         
    plt.plot(x,y)    
    return 0

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = stochastic_gradient_descent(X_train_scale, y_train)
    y_hat = predict(X_train_scale, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))
    plot_roc_curve(y_hat,y_test)

if __name__ == "__main__":
    main(sys.argv)


