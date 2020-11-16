# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:18:44 2020

@author: ahadj
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
from tqdm import tqdm
import copy

def round_down(n, decimals=0):#used to identify in which cell a node belongs to
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def GenerateCoorArray(N,x_size,y_size):#prepares a list with the coordinates of all nodes
    pmcell=int(N/10)#number of pacemaker cells
    CoorStore = []
    for i in range(0,pmcell):#sets the first 10% of nodes as pacemakers
        Coor = []
        Coor.append(0)
        Coor.append(random.random()*y_size)
        CoorStore.append(Coor) 
    for i in range(pmcell,N):   
        Coor = []
        Coor.append(random.random()*x_size)
        Coor.append(random.random()*y_size)
        CoorStore.append(Coor) 
    return sorted(CoorStore , key=lambda k: [k[0], k[1]])

def distance_checker(R,i,j,CoorStore,y_size):#checks the distance between 2 nodes 
    #in a way to allow periodic BCs in y-direction and open in x-direction
    diff=CoorStore[i][1]-CoorStore[j][1]
    if abs(diff)<=R:# for rest cases
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    elif abs(diff)>R:    
        if CoorStore[i][1]+R>y_size:#sets upper BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]-y_size - CoorStore[j][1]) **2 )**0.5
        elif CoorStore[i][1]-R<0:#sets lower BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]+y_size - CoorStore[j][1]) **2 )**0.5
        else:
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    if d <= R:
        return True
    else:
        return False       

def ConnectionArray(N,x_size,y_size,CoorStore,R,t_memory):
    #a list containing all the connections of the ith node in list
    t_memory_array = np.zeros(t_memory)
    ConnectionsStore = []
    ChargeFlowStore = []
    
    for i in range(N):
        connections = []
        ChargeFlow = []
        nodecoor=CoorStore[i]
        ll=nodecoor[0]-R
        rl=nodecoor[0]+R
        for j in range(N):
            if i !=j and ll<=CoorStore[j][0]<=rl:    
                if distance_checker(R,i,j,CoorStore,y_size) == True:
                    connections.append(j)
                    ChargeFlow.append(np.copy(t_memory_array))
        ConnectionsStore.append(connections)
        ChargeFlowStore.append(ChargeFlow)
        
    ChargeFlowOUTstore = copy.deepcopy(ChargeFlowStore)
    ChargeFlowINstore = copy.deepcopy(ChargeFlowStore)
    return ConnectionsStore, ChargeFlowINstore, ChargeFlowOUTstore

def FunctionalArray(N,x_size,y_size,tau,delta,epsilon):
    #prepares two lists: one with refractory period and one with excitation
    #probability of each node
    taustore = []
    epsilonstore = []
    for i in range(N):
        taustore.append(tau)
        if random.random() < delta:
            epsilonstore.append(epsilon)
        else:
            epsilonstore.append(0)
    functionalstore  =[]
    functionalstore.append(taustore)
    functionalstore.append(epsilonstore)
    return functionalstore
 
def GenerateState(N):#prepares list with state of each node
    return [0]*N
#%%      
def PaceMaker(StateListP,FunctionListP,N):#excites pacemaker cells every T time steps
    pmcell=int(N/10)
    for i in range(pmcell):
        rand=random.random()
        if rand>FunctionListP[1][i]:
            StateListP[i]= FunctionListP[0][i] + 1                
    return StateListP

def UpdateStateLattice(N, StateListU, FunctionListU, Connection,ChargeFlowIn,ChargeFlowOut):
#updates the state of each node
  
    ChargeFlowIN=copy.deepcopy(ChargeFlowIn)
    ChargeFlowOUT = copy.deepcopy(ChargeFlowOut)
    
    Updatedlist = [0]* N
    for i in range(N):
        if StateListU[i] > 0: #reduce refractory period countdown by one
            Updatedlist[i] = StateListU[i]- 1            
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]
                ChargeFlowIn,ChargeFlowOut = ChargeFlowUpdate(ChargeFlowIN, ChargeFlowOUT,i,j,NeighNode,False,Connection)               
            #continue
        if StateListU[i]== 0: #here check if any neighbours are in firing state
            rand=random.random()
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]
                if StateListU[NeighNode] == FunctionListU[0][NeighNode]+1:
                    if rand>FunctionListU[1][NeighNode]:
                        Updatedlist[i]=FunctionListU[0][i]+1
                        ChargeFlowIn,ChargeFlowOut = ChargeFlowUpdate(ChargeFlowIN, ChargeFlowOUT,i,j,NeighNode,True,Connection)
                else:
                    ChargeFlowIn,ChargeFlowOut = ChargeFlowUpdate(ChargeFlowIN, ChargeFlowOUT,i,j,NeighNode,False,Connection)
                    
    return Updatedlist,ChargeFlowIn, ChargeFlowOut
 
def ChargeFlowUpdate(ChargeFlowIn, ChargeFlowOut,i,j,NeighNode,state,Connection):

    if state == True:
        ChargeFlowIn[i][j] = np.insert(ChargeFlowIn[i][j],0,1) #insert the first element with updated
        ChargeFlowIn[i][j] = ChargeFlowIn[i][j][:-1] #remove the last element ie getting rid of the oldest value
        
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = np.insert(ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)],0,1)
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)][:-1]
        
    if state == False:
        ChargeFlowIn[i][j] = np.insert(ChargeFlowIn[i][j],0,0) #replace the first element with updated
        ChargeFlowIn[i][j] = ChargeFlowIn[i][j][:-1] #remove the last element ie getting rid of the oldest value
        
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = np.insert(ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)],0,0)
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)][:-1]
        
    return ChargeFlowIn,ChargeFlowOut

#%%
def RunState(TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T,t_memory):
    #runs the model for a specified number of timesteps
    #and returns a list with all timesteps
    SetupS=timeit.default_timer()
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections, ChargeFlowIn, ChargeFlowOut = ConnectionArray(N,x_size,y_size,Coordinates,r,t_memory)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateListR = GenerateState(N)
    SetupE=timeit.default_timer()
    
    StateStore = []
    
    ChargeFlowINAll=[]
    ChargeFlowOUTALL = []
    
    RunS=timeit.default_timer()
    for i in range(TimeSteps):
        print(i)
        if i == 0 or i % T == 0:          
            StateListR = PaceMaker(StateListR,FunctionListR,N)
            StateListR, ChargeFlowIn, ChargeFlowOut = UpdateStateLattice(N,StateListR,FunctionListR,Connections,ChargeFlowIn,ChargeFlowOut)
            StateStore.append(StateListR)
            ChargeFlowINAll.append(ChargeFlowIn)
            ChargeFlowOUTALL.append(ChargeFlowOut)
 
        else:
            StateListR, ChargeFlowIn, ChargeFlowOut = UpdateStateLattice(N,StateListR,FunctionListR,Connections,ChargeFlowIn,ChargeFlowOut)
            StateStore.append(StateListR)
            ChargeFlowINAll.append(ChargeFlowIn)
            ChargeFlowOUTALL.append(ChargeFlowOut)
             
    RunE=timeit.default_timer()    
    TimeS=SetupE-SetupS
    TimeR=RunE-RunS
    data=[TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T]
    return StateStore, Coordinates, Connections,TimeS,TimeR, ChargeFlowINAll, ChargeFlowOUTALL, TimeSteps,data
 
def Resultant_Vectors(ChargeFlowINALL,ChargeFlowOUTALL, Connections, Coordinates,TimeSteps,R,y_size,outv=False,inv=False):
    ChargeFlowINsingle = copy.deepcopy(ChargeFlowINALL)
    ChargeFlowOUTsingle = copy.deepcopy(ChargeFlowOUTALL)
    
    ChargeFlowResultant = copy.deepcopy(ChargeFlowINALL)
    
    for t in range(TimeSteps):
        for i in range(len(Connections)):         
            for j in range(len(Connections[i])): #for each time step calcualte the memory avergae
                ChargeFlowINsingle[t][i][j] = np.mean(ChargeFlowINALL[t][i][j])
                ChargeFlowOUTsingle[t][i][j] = np.mean(ChargeFlowOUTALL[t][i][j])
                            
    for t in range(TimeSteps):
        for i in range(len(Connections)):
            INvector = [0,0]
            OUTvector = [0,0]
            #so that the resulting vector is the average
            totchargeIn=sum(ChargeFlowINsingle[t][i])
            totchargeOut=sum(ChargeFlowOUTsingle[t][i])
            for j in range(len(Connections[i])):
                if totchargeIn!=0:                    
                    ucomIn=Coordinates[i][0] - Coordinates[Connections[i][j]][0]
                    INvector[0] += (ucomIn*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                    vcomIn=Coordinates[i][1] - Coordinates[Connections[i][j]][1]                                        
                    if abs(vcomIn)<=R:                                                                        
                        INvector[1] += (vcomIn*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                    elif abs(vcomIn)>R:
                        if vcomIn+R>y_size:                            
                            INvector[1] += ((vcomIn-y_size)*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                        elif vcomIn<0:                            
                            INvector[1] += ((vcomIn+y_size)*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                                            
                if totchargeOut!=0:
                    ucomOut=Coordinates[Connections[i][j]][0] - Coordinates[i][0]
                    OUTvector[0] += (ucomOut*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                    vcomOut=Coordinates[Connections[i][j]][1] - Coordinates[i][1]
                    if abs(vcomOut)<=R:                        
                        OUTvector[1] += (vcomOut*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                    elif abs(vcomOut)>R:
                        if vcomOut+R>y_size:                            
                            OUTvector[1] += ((vcomOut-y_size)*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                        elif vcomOut<0:
                            OUTvector[1] += ((vcomOut+y_size)*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))

            if outv==False and inv==False:
                ChargeFlowResultant[t][i] = [(INvector[0]+OUTvector[0])/2,(INvector[1]+OUTvector[1])/2]
            if outv==True:
                ChargeFlowResultant[t][i]=[OUTvector[0],OUTvector[1]]
            if inv==True:
                ChargeFlowResultant[t][i]=[INvector[0],INvector[1]]
    return ChargeFlowResultant #this is hopefully a list arranged first by time, and then each ndoe given a vector in the form of [x,y]       
            
def MovieNodes(a):#input is list with all the lists describing the state of nodes(StateStore)
    #prepares movie for evolution of state of nodes
    x=[]
    y=[]
    for i in range(len(a[1])):
        x.append(a[1][i][0])
        y.append(a[1][i][1])
    fig=plt.figure()
    ims=[]
    for i in range(len(a[0])):
        im=plt.scatter(x,y,c=a[0][i],edgecolors='r',cmap=plt.cm.binary)
        ims.append([im])
    plt.colorbar()
    ani = animation.ArtistAnimation(fig, ims, interval=500, 
                                     repeat_delay=1000)

    return ani

def Pixelation(cc,x_grid_size,y_grid_size):
    #prepares pixelated movie based on resolution requested
    x_size=cc[8][2]
    y_size=cc[8][3]
    tau=cc[7][4]
    grid_coor = []
    for i in range(int(x_grid_size)):
        for j in range(int(y_grid_size)):
            grid_coor.append([i,j])
    grid_container = []
    timeseries=[]#contains time-series for each cell
    for i in range(len(grid_coor)):
        grid_container.append([])
        timeseries.append([])
    for i in range(len(cc[1])):
        grid_coor_state = round_down(cc[1][i][0]/(x_size/x_grid_size)), round_down(cc[1][i][1]/(y_size/y_grid_size)) 
        grid_container[int(grid_coor_state[0]*(x_grid_size) + grid_coor_state[1] )].append(i)
    allgridvalues=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum = 0
            for node in range(len(grid_container[cell])):
                sum = sum + cc[0][i][grid_container[cell][node]]
            grid_sum[y_grid_size-1-grid_coor[cell][1]][grid_coor[cell][0]] = sum
            timeseries[cell].append(sum)
        allgridvalues.append(grid_sum)                
    nodespc=[]#nodespercell(determining cell with max number of nodes)
    for i in range(len(grid_container)):
        nodespc.append(len(grid_container[i]))
    maxcellcolor=np.mean(nodespc)*(tau+1)#determining max value possible 
    #in grid_sum,required to set the color scale
    return allgridvalues,int(maxcellcolor) ,timeseries

def MoviePixels(pixeldata):
    Allgridvalues=pixeldata[0]
    fig = plt.figure()
    ims=[]    
    for i in range(len(Allgridvalues)):
        im=plt.imshow(Allgridvalues[i],interpolation='none',cmap=plt.cm.binary,vmin=0,vmax=pixeldata[1],animated=True)
        if i==0:
            fig.colorbar(im)
        ims.append([im])
    plt.title('Pixelated Grid')
    ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                    repeat_delay=1000)
    return ani
#%%doing a speed test
#compared np.array and list method for out and in vector and decided
#to use np.array(slightly faster)
speedT = RunState(20,8000,128,128,5,1,0.22,4,20,3)
print(speedT[3],speedT[4])
#%%
x = RunState(50,4000,90,90,10,1,0.6,5,30,3)
#%%
MovieNodes(x)
#%%
Vectors = Resultant_Vectors(x[5],x[6],x[2],x[1],x[7],5,90)
#%%
X=[]
Y=[]
U=[]
V=[]
for i in range(len(x[1])):
    X.append(x[1][i][0])
    Y.append(x[1][i][1])
    U.append(0)
    V.append(0)
fig, ax = plt.subplots(1,1)
Q = ax.quiver(X,Y,U, V, pivot='tail', angles='uv', scale_units='xy',scale=2)

def update_quiver(num,Q):
    U=[0]*len(x[1])
    V=[0]*len(x[1])
    Q.set_UVC(U,V)
    for i in range(len(x[1])):
        U[i]=Vectors[num][i][0]
        V[i]=Vectors[num][i][1]
    Q.set_UVC(U,V)
    return Q,

anim = animation.FuncAnimation(fig, update_quiver,frames=len(Vectors),fargs=(Q,),
                               interval=500, blit=True)
fig.tight_layout()
#%%
def ArrowFrameMicro(num,Q):
    U=[0]*len(x[1])
    V=[0]*len(x[1])
    Q.set_UVC(U,V)
    for i in range(len(x[1])):
        U[i]=Vectors[num][i][0]
        V[i]=Vectors[num][i][1]
    Q.set_UVC(U,V)
    return Q,
#%%
X=[]
Y=[]
U=[]
V=[]
for i in range(len(x[1])):
    X.append(x[1][i][0])
    Y.append(x[1][i][1])
    U.append(0)
    V.append(0)
fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='tail', angles='uv', scale_units='xy',scale=2)
ArrowFrameMicro(21,Q)