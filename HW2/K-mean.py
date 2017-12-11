# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:55:55 2017

@author: Sid007
"""

import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

cl=np.zeros(150)
iris = load_iris()
list = np.c_[iris.data]
data = list[:,2:]




def distance(vector1,vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))

def initcenter(k,dataset):
    
    center=np.zeros((k,2))
    for i in range(k):  
        index = int(random.uniform(0, 150))  
        center[i] = data[index]  
    return center

def K_mean(k,data):
    
   ct = initcenter(k,data)
   
   for i in range (0,k):
       for j in range(i+1,k):
          if np.array_equal(ct[i],ct[j]):
              ct = initcenter(k,data)
              
   ct_old=np.zeros((k,2))
   
   change=0
   
   while change == 0:
       
       Sum=np.zeros((k,2)) 
       count=np.zeros((k,1))
       
       for i in range (0,150):
           mindis = 10.0
           for j in range (0,k):
               dis = distance(ct[j,:],data[i,:])
               if dis < mindis:
                   mindis = dis
                   cl[i] = j
                     
               
    
       for i in range (0,150):
           for j in range (0,k):
               if cl[i] == j:
                   Sum[j,:]=data[i,:]+Sum[j,:]
                   count[j]=count[j]+1
                   
       for i in range (0,k):
           ct_old[i]=ct[i]
           ct[i]=Sum[i]/count[i]
       
       if np.array_equal(ct,ct_old):
           change=1
          
   for i in range (0,150):
        if cl[i]==0:
            plt.scatter(data[i,0],data[i,1], c='c',  marker='s', alpha=.4)
        elif cl[i]==1:
            plt.plot(data[i,0],data[i,1], c='g', marker='^', alpha=.4)
        elif cl[i]==2:
            plt.plot(data[i,0],data[i,1], c='r', marker='*', alpha=.4)
        elif cl[i]==3:
            plt.plot(data[i,0],data[i,1], c='k', marker='8', alpha=.4)
        elif cl[i]==4:
            plt.plot(data[i,0],data[i,1], c='m', marker='X', alpha=.4)
                 
    
   plt.show()
    
for i in range (2,6):
    K_mean(i,data)
    

