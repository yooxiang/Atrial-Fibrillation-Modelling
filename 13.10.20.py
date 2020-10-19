# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:21:56 2020

@author: ahadj
"""
"""improvement for up and down directions to have finite probability 
firing"""

import scipy as sp
import numpy as np
import random
from scipy import linalg
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation



#function for generating the structure
#if entry is 0 there is no connection transversally
def GenerateStructure(x,y,v):
    Empty=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            rand=random.random()
            if rand<v:
                Empty[i][j]=1
    return Empty
#function below prepares matrix with functional properties of cells
#one matrix has refractory period of cells
#other has probability of a cell not exciting when required
def GenerateFunction(x,y,tau,delta,epsilon):
    Empty_tau=np.zeros((x,y))
    Empty_epsilon=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            rand=random.random()
            if rand<delta:
                epsilon_i=epsilon
            else:
                epsilon_i=0
            Empty_tau[i][j]=tau
            Empty_epsilon[i][j]=epsilon_i
    return Empty_tau,Empty_epsilon
             
def GenerateState(x,y):
    Empty=np.zeros((x,y))
    return Empty
#%%
def PaceMaker(StateLattice,FunctionLattice, x_size, y_size):
    for i in range(y_size):
        rand=random.random()
        if rand>FunctionLattice[1][i][0]:
            StateLattice[i][0] = FunctionLattice[0][i][0] + 1                
    return StateLattice
            
def BoundaryChecker(i,j, y_size):
    if i == y_size - 1:
        return -1 
    else:
        return i
            
def UpdateStateLattice(StructureLattice,FunctionLattice, StateLattice,x_size,y_size):
  
    UpdatedArray = np.zeros((y_size,x_size))
    for i in range(y_size):
        for j in range(x_size):
           
            if StateLattice[i][j] > 0: #reduce refractory period countdown by one
                UpdatedArray[i][j] = StateLattice[i][j] - 1
                continue
                
            if StateLattice[i][j] == 0: #here check if any neighbours are in firing state
                rand=random.random()
                if j-1 != -1:#left boundary conditions
                    if StateLattice[i][j - 1] == FunctionLattice[0][i][j-1] + 1 : #left
                        if rand>FunctionLattice[1][i][j-1]:                            
                            UpdatedArray[i][j] = FunctionLattice[0][i][j] + 1                
                            continue
                if j+1 != x_size:#right boundary conditions
                    if StateLattice[i][j + 1 ] == FunctionLattice[0][i][j +1 ] + 1: #right
                        if rand>FunctionLattice[1][i][j+1]:
                            UpdatedArray[i][j] = FunctionLattice[0][i][j] + 1 
                            continue

                if StructureLattice[i][j] == 1 and StateLattice[i-1][j] == FunctionLattice[0][i-1][j] + 1: #up
                    if rand>FunctionLattice[1][i-1][j]:
                        UpdatedArray[i][j] = FunctionLattice[0][i][j] + 1   
                        continue            
                i = BoundaryChecker(i,j,y_size) #down
             
                if StructureLattice[i+1][j] == 1 and StateLattice[i+1][j] == FunctionLattice[0][i+1][j] + 1:
                    if rand>FunctionLattice[1][i+1][j]:
                        UpdatedArray[i][j] = FunctionLattice[0][i][j] + 1 
                                
    return UpdatedArray

def RunState(TimeSteps,x_size,y_size,tau,delta,epsilon,v,T):
    StructureLattice = GenerateStructure(x_size,y_size,v)
    FunctionLattice = GenerateFunction(x_size,y_size,tau,delta,epsilon)
    StateLattice = GenerateState(x_size,y_size)
    
    LatticeStore = []

    for i in range(TimeSteps):
        
        if i == 0 or i % T == 0:
            StateLattice = PaceMaker(StateLattice,FunctionLattice,x_size,y_size)
            LatticeStore.append(StateLattice)
        else:
            StateLattice = UpdateStateLattice(StructureLattice,FunctionLattice,StateLattice,x_size,y_size)
            LatticeStore.append(StateLattice)
    return LatticeStore, StructureLattice, FunctionLattice

#%%
#trying to visualise the project
fig = plt.figure()

p=RunState(400,100,100,5,0.1,0.3,0.9,20)
ims=[]
for i in range(len(p[0])):
    im=plt.imshow(p[0][i],interpolation='none',cmap=plt.cm.binary,animated=True)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=30, 
                                repeat_delay=1000)
