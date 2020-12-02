# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:30:09 2020

@author: ahadj
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
import copy
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def round_down(n, decimals=0):#used to identify in which cell a node belongs to
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def GenerateCoorArray(N,x_size,y_size):#prepares a list with the coordinates of all nodes
    pmcell=int(N/10)#number of pacemaker cells
    CoorStore =[]
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
            avgcheck=False
            #so that the resulting vector is the average
            if outv==False and inv==False and t!=0:
                totchargeIn=sum(ChargeFlowINsingle[t-1][i])
                avgcheck=True
            else: 
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
                #condition for taking the average between the previous time step of in vector
                if totchargeIn!=0 and avgcheck==True:                    
                    ucomIn=Coordinates[i][0] - Coordinates[Connections[i][j]][0]
                    INvector[0] += (ucomIn*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))
                    vcomIn=Coordinates[i][1] - Coordinates[Connections[i][j]][1]                                        
                    if abs(vcomIn)<=R:                                                                        
                        INvector[1] += (vcomIn*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))
                    elif abs(vcomIn)>R:
                        if vcomIn+R>y_size:                            
                            INvector[1] += ((vcomIn-y_size)*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))
                        elif vcomIn<0:                            
                            INvector[1] += ((vcomIn+y_size)*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))                

            if outv==False and inv==False:
                #print('hereavg')
                ChargeFlowResultant[t][i] = [(INvector[0]+OUTvector[0])/2,(INvector[1]+OUTvector[1])/2]
            if outv==True:
                mag= np.sqrt((OUTvector[0])**2+(OUTvector[1])**2)
                if mag != 0:
                    finalvector=[OUTvector[0]/mag,OUTvector[1]/mag]
                else:
                    finalvector=[OUTvector[0],OUTvector[1]]
                ChargeFlowResultant[t][i]=finalvector
            if inv==True:
                mag= np.sqrt((INvector[0])**2+(INvector[1])**2)
                if mag != 0:
                    finalvector=[INvector[0]/mag,INvector[1]/mag]
                else:
                    finalvector=[INvector[0],INvector[1]]                
                ChargeFlowResultant[t][i]=finalvector
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
    #prepares pixelation of nodes based on resolution requested
    #cc is output of RunState
    x_size=cc[8][2]
    y_size=cc[8][3]
    tau=cc[8][4]
    grid_coor = []
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    grid_container = []
    timeseries=[]#contains time-series for each cell
    for i in range(len(grid_coor)):
        grid_container.append([])
        timeseries.append([])
    for i in range(len(cc[1])):
        grid_coor_state = cc[1][i][0]//(x_size/x_grid_size),cc[1][i][1]//(y_size/y_grid_size) 
        grid_container[int(grid_coor_state[1]*(x_grid_size) + grid_coor_state[0] )].append(i)
    allgridvalues=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum_c = 0
            for node in range(len(grid_container[cell])):
                sum_c += cc[0][i][grid_container[cell][node]]
            grid_sum[y_grid_size-1-grid_coor[cell][1]][grid_coor[cell][0]]=sum_c
            timeseries[cell].append(sum_c)
        allgridvalues.append(grid_sum)                
    nodespc=[]#nodespercell(determining cell with max number of nodes)
    for i in range(len(grid_container)):
        nodespc.append(len(grid_container[i]))
    maxcellcolor=np.mean(nodespc)*(tau+1)#determining max value possible 
    #in grid_sum,required to set the color scale
    return allgridvalues,int(maxcellcolor) ,timeseries,grid_container,grid_coor

def MoviePixels(pixeldata):
    #input is output of Pixelation
    Allgridvalues=pixeldata[0]
    fig = plt.figure()
    ims=[]    
    for i in range(len(Allgridvalues)):
        im=plt.imshow(Allgridvalues[i],interpolation='none',cmap=plt.cm.binary,vmin=0,vmax=pixeldata[1],animated=True)
        if i==0:
            fig.colorbar(im)
        ims.append([im])
    plt.title('Pixelated Grid')
    ani = animation.ArtistAnimation(fig, ims, interval=500, 
                                    repeat_delay=1000)
    return ani

def PixelatedVectors(cc,cv,x_grid_size,y_grid_size):
    #prepares exact coarse graining
    #cc is output of Runstate
    #cv is output of resultant vector
    x_size=cc[8][2]
    y_size=cc[8][3]
    grid_coor = []#cotains coordinates for each pixel
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    grid_container = []#contains nodes in each pixel
    for i in range(len(grid_coor)):
        grid_container.append([])
    for i in range(len(cc[1])):
        grid_coor_state = cc[1][i][0]//(x_size/x_grid_size),cc[1][i][1]//(y_size/y_grid_size) 
        grid_container[int(grid_coor_state[1]*(y_grid_size) + grid_coor_state[0] )].append(i)
    
    gridvectorTot=[]
    for t in range(len(cv)):
        gridvector_t=[]
        for cell in range(len(grid_container)):
            cellvector=[0,0]
            no_nodes=len(grid_container[cell])
            for i in range(no_nodes):
                cellvector[0]+=cv[t][grid_container[cell][i]][0]/no_nodes
                cellvector[1]+=cv[t][grid_container[cell][i]][1]/no_nodes
            cellvector_mag=np.sqrt((cellvector[0])**2+(cellvector[1])**2)
            if cellvector_mag!=0:
                cellvector_norm=[cellvector[0]/cellvector_mag,cellvector[1]/cellvector_mag]
            else:
                cellvector_norm=cellvector
            gridvector_t.append(cellvector_norm)
        gridvectorTot.append(gridvector_t)                       
    return gridvectorTot,grid_coor
#%%
def divergence(vector,Coordiantes,x_dim,y_dim): #x_dim and y_dim are nunber of cells, not the true length which is captured in the coordinates
    
    def internal_point(index): # this is for all the points apart from the edges on the left and right
        #vector = vector_field[index]
        y_coor =(index//x_dim)        
        dFx = (vector[index+1][0] - vector[index-1][0])/(Coordiantes[index+1][0] - Coordiantes[index-1][0])
                
        if y_coor == 0:
            dFy = (vector[index+x_dim][1] - vector[index+(x_dim*(y_dim -1))][1])/(2*(Coordiantes[index+x_dim][1] - Coordiantes[index][1]))
        if y_coor == y_dim - 1:   
            dFy = (vector[index-(x_dim*(y_dim -1))][1] - vector[index - x_dim][1])/(2*(Coordiantes[index][1] - Coordiantes[index-x_dim][1]))
        else:
            dFy = (vector[index+x_dim][1] - vector[index - x_dim][1])/(Coordiantes[index+x_dim][1] - Coordiantes[index-x_dim][1])         
        return dFx + dFy   
    
    def edge_point(index): #this is for all the points on the right and left ie x = 0 or x= y_di
        #here a central difference is not used
        #vector = vector_field[index]
        x_coor = index % y_dim 
        y_coor =(index//x_dim)
        
        if x_coor == 0:          
            dFx = (vector[index+1][0] - vector[index][0])/(Coordiantes[index+1][0] - Coordiantes[index][0])            
        if x_coor == y_dim - 1:
            dFx = (vector[index][0] - vector[index-1][0])/(Coordiantes[index][0] - Coordiantes[index-1][0])            
        if y_coor == 0:        
            dFy = (vector[index+x_dim][1] - vector[index+(x_dim*(y_dim -1))][1])/(2*(Coordiantes[index+x_dim][1] - Coordiantes[index][1]))
        if y_coor == y_dim - 1:
            dFy = (vector[index-(x_dim*(y_dim -1))][1] - vector[index - x_dim][1])/(2*(Coordiantes[index][1] - Coordiantes[index-x_dim][1]))            
        if y_coor > 0 and y_coor < y_dim - 1:
            dFy = (vector[index+x_dim][1] - vector[index - x_dim][1])/(Coordiantes[index+x_dim][1] - Coordiantes[index-x_dim][1])
        
        return dFx + dFy

    div_store = []
    for index in range(x_dim*y_dim):
        x_coor = index % y_dim 

        if x_coor > 0 and x_coor < y_dim -1:
            div = internal_point(index)
        else:
            div = edge_point(index)
        div_store.append(div)
    
    return div_store,Coordiantes
#following function creates vector movies
def VectorMovie(vectordata,points,frame):
    #vectordata must be the vector of either nodes or pixels
    #with their respective points
    #if frame==None then it returns the whole movie otherwise specify
    #the frame you need to visualise
    X=[i[0] for i in points]
    Y=[i[1] for i in points]
    U=[0]*len(points)
    V=[0]*len(points)
    
    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X,Y,U, V, pivot='tail', angles='xy', scale_units='xy',scale=1)
    
    def update_quiver(num,Q):
        U=[0]*len(points)
        V=[0]*len(points)
        Q.set_UVC(U,V)
        for i in range(len(points)):
            U[i]=vectordata[num][i][0]
            V[i]=vectordata[num][i][1]
        colors = np.arctan2(np.array(V),np.array(U))
        #norm = Normalize()
        #norm.autoscale(colors)
        #Q.set_UVC(U,V,cm.twilight_shifted(colors))
        Q.set_UVC(U,V)
        return Q,
    
    if frame==None:
        anim1 = animation.FuncAnimation(fig, update_quiver,frames=len(vectordata),fargs=(Q,), interval=500, blit=True)
    else:
        anim1=update_quiver(frame,Q)
    fig.tight_layout()
    
    return anim1
#function that returns movie of divergence
def DivMovie(vectordata,coord,frame_no):
    #input vectordata is the vectors of each pixel
    #coord is the coordinates of the pixels
    #if frame_no==None then it returns a Movie
    x_s=max([i[0] for i in coord])+1#sets x length of grid
    y_s=max([i[1] for i in coord])+1#sets y length of grid
    div_all=[]#list with divergence data for all times
    div_max=[]
    div_min=[]
    for i in range(len(vectordata)):
        div_loop=divergence(vectordata[i],coord,y_s,x_s)
        div_all.append(div_loop[0])
        div_max.append(max(div_loop[0]))
        div_min.append(min(div_loop[0]))
    #frame_all contains the frames for all time steps
    frame_all=[]
    for time in range(len(div_all)):
        frame_i=np.zeros([y_s,x_s])
        for cell in range(len(coord)):
            frame_i[y_s-1-coord[cell][1]][coord[cell][0]]=div_all[time][cell]
        frame_all.append(frame_i)
        
    fig = plt.figure() 
    
    if frame_no==None:
        ims=[]    
        for i in range(len(frame_all)):
            im=plt.imshow(frame_all[i],interpolation='none',cmap='jet',vmin=min(div_min),vmax=max(div_max),animated=True)
            if i==0:
                fig.colorbar(im)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=500,repeat_delay=1000)
    
    else:
        ani=plt.imshow(frame_all[frame_no],interpolation='none',cmap='jet',vmin=min(div_min),vmax=max(div_max),animated=True)
        fig.colorbar(ani)
    plt.title('Pixelated Grid')
    
    return ani
#%%
def ReentrantDot(vectordata,points,xdim,ydim):
    avgdot_all=[]
    for time in range(len(vectordata)):
        avgdot_time=[]
        for i in range(len(vectordata[time])):
            x_loc=i%xdim#find x coord. of respective pixel
            y_loc=i//xdim#finds y coord. of respective pixel
            surrvec=[]#surrounding vectors
            
            if x_loc==(xdim-1):#right BC
                right=0
            else:
                right=sum([vectordata[time][i][j]*vectordata[time][i+1][j] for j in range(len(vectordata[time][i]))])
                surrvec.append(right)
            if x_loc!=0:#left BC
                left=sum([vectordata[time][i][j]*vectordata[time][i-1][j] for j in range(len(vectordata[time][i]))])
                surrvec.append(left)
            else:
                left=0
            if y_loc==(ydim-1):# up BC
                up=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)][j] for j in range(len(vectordata[time][i]))])
            else:
                up=sum([vectordata[time][i][j]*vectordata[time][i+xdim][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(up)
            if y_loc==0:#down BC
                down=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)][j] for j in range(len(vectordata[time][i]))])
            else:    
                down=sum([vectordata[time][i][j]*vectordata[time][i-xdim][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(down)
            #below are diagonal terms
            if x_loc==xdim-1:#up right diag
                uprightdiag=0
            else:
                if y_loc==(ydim-1):
                    uprightdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)+1][j] for j in range(len(vectordata[time][i]))])
                else:
                    uprightdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim+1][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(uprightdiag)
            if x_loc==(xdim-1):#down right diag
                downrightdiag=0
            else:
                if y_loc==0:
                    downrightdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)+1][j] for j in range(len(vectordata[time][i]))])
                else:
                    downrightdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+1][j] for j in range(len(vectordata[time][i]))])                    
            surrvec.append(downrightdiag)
            if x_loc==0:#up left diag
                upleftdiag=0
            else:
                if y_loc==(ydim-1):
                    upleftdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)-1][j] for j in range(len(vectordata[time][i]))])
                else:
                    upleftdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-1][j] for j in range(len(vectordata[time][i]))])            
            surrvec.append(upleftdiag)
            if x_loc==0:#down left diag
                downleftdiag=0
            else:
                if y_loc==0:
                    downleftdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)-1][j] for j in range(len(vectordata[time][i]))])
                else:
                    downleftdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim-1][j] for j in range(len(vectordata[time][i]))]) 
            surrvec.append(downleftdiag)
            
            surrvecN=len(surrvec)
            sc=0
            for i in range(len(surrvec)):
                if surrvec[i]==0:
                    sc+=1
            #print(sc,surrvecN)
            if sc==surrvecN:
                avgdot=None
            else:
                avgdot=sum(surrvec)/surrvecN
            avgdot_time.append(avgdot)
        
        avgdot_all.append(avgdot_time)
    
    return avgdot_all
#%%
def divergence_histogram(vector_data, coord, log_scale = False, curve_fitting = False):
    x_size=max([i[0] for i in coord])+1 #sets x length of grid
    y_size =max([i[1] for i in coord])+1 #sets y length of grid
    div_all=[]#list with divergence data for all times
    for time_step in range(len(vector_data)):
        divergence_matrix = divergence(vector_data[time_step],coord,y_size,x_size)
        for cell in range(len(divergence_matrix[0])):
            div_all.append(divergence_matrix[0][cell])
    
    div_all.sort()
    N = len(div_all)

    def percentile(p,N,div_all):
       #plt.axvline(x=div_all[round((N*p)/100)], label = p)
       #plt.axvline(x=div_all[ N - round((N*p)/100)], label = str(p) + 'th percentile', linestyle = 'dashed')
       return None
     
    percentile(5, N, div_all)
    percentile(1, N, div_all)
    percentile(0.1, N, div_all)
    
    counts,bin_edges = np.histogram(div_all,20)
    counts=[1 if i==0 else i for i in counts]#for now to deal with log10(0)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    
    if curve_fitting == False:
        plt.scatter(bin_centres, counts, marker = 'x')
        if log_scale == True:
            plt.yscale('log')
        
    if curve_fitting == True:
        
        def gaussian_function(x,omega,a):
            return a*(1/omega*np.sqrt(2*np.pi))*np.exp(-0.5*(x/omega)**2)
        
        bin_smooth = np.linspace(min(bin_centres)-0.5,max(bin_centres)+0.5,1000)
        popt, pcov = curve_fit(gaussian_function,bin_centres,np.log10(counts))
        plt.scatter(bin_centres,np.log10(counts), marker = 'x', label = 'data')
        plt.plot(bin_smooth, gaussian_function(bin_smooth, *popt), label = "Gaussian Fit with omega =" + str(popt[0])+r'$\pm$'+str(np.sqrt(pcov[0,0])))
        
    plt.xlabel("Divergence")
    plt.ylabel("Frequency")
    #plt.legend()
    plt.show()
    print("Gaussian Fit with omega =" + str(popt[0])+r'$\pm$'+str(np.sqrt(pcov[0,0])))
    print('Alpha coefficient is',str(popt[1]),'+-',str(np.sqrt(pcov[1,1])) )
    return [popt[1],np.sqrt(pcov[1,1])]
    
def divergence_map(vector_data, coord, percentile):
    x_size=max([i[0] for i in coord])+1 #sets x length of grid
    y_size =max([i[1] for i in coord])+1 #sets y length of grid
    div_all=[]#list with divergence data for all times
    for time_step in range(len(vector_data)):
        divergence_matrix = divergence(vector_data[time_step],coord,y_size,x_size)
        for cell in range(len(divergence_matrix[0])):
            div_all.append(divergence_matrix[0][cell])
        
    divergence_heat_map = [0]*len(coord)    
    div_all_sorted = copy.deepcopy(div_all)    
    div_all_sorted.sort()
    N = len(div_all)
    
    starting_index = N - round((N*percentile)/100) 
    
    def index_return(index):
        t = index//(x_size*y_size)
        return index - t*(x_size*y_size)
    
    for i in range(N-starting_index):
        index = div_all.index(max(div_all))
        divergence_heat_map[index_return(index)] += 1
        div_all[index] = 0

    divergence_heat_map_matrix = []
    for i in range(x_size):
        divergence_heat_map_matrix.append([])
        for j in range(y_size):
            divergence_heat_map_matrix[i].append(divergence_heat_map[(i*x_size)+j])    
     
    fig=plt.figure()    
    im=plt.imshow(divergence_heat_map_matrix,interpolation='none',cmap='binary',vmin=0,animated=True)    
    plt.gca().invert_yaxis()
    fig.colorbar(im)    
    return im
#following function identifies and returns location of AF phenomena
#and their respective time
def AFIdentify(vector_data, coord, percentile):
    x_size=max([i[0] for i in coord])+1 #sets x length of grid
    y_size =max([i[1] for i in coord])+1 #sets y length of grid
    div_all=[]#list with divergence data for all times
    for time_step in range(len(vector_data)):
        divergence_matrix = divergence(vector_data[time_step],coord,y_size,x_size)
        for cell in range(len(divergence_matrix[0])):
            div_all.append(divergence_matrix[0][cell])
    
    N = len(div_all)
    starting_index = N - round((N*percentile)/100) 
    
    def index_return(index):
        t = index//(x_size*y_size)
        return index - t*(x_size*y_size),t
    
    AFinc=[]
    for i in range(N-starting_index):
        index = div_all.index(max(div_all))
        div_all[index] = 0
        AFd=index_return(index)
        AFloc=[AFd[0]%x_size,AFd[0]//x_size]
        AFinc.append([AFloc,AFd[1]])    
    return AFinc
 
def HistoMovie(vectordata,coord,frame_no):
    x_s=max([i[0] for i in coord])+1#sets x length of grid
    y_s=max([i[1] for i in coord])+1#sets y length of grid
    div_all=[]#list with cumulative divergence data for all times
    for i in range(len(vectordata)):
        div_loop=divergence(vectordata[i],coord,y_s,x_s)
        if i==0:
            div_all.append(div_loop[0])
        else:
            div_all.append(div_all[-1]+div_loop[0])   
    
    def update_quiver(num,data):
        plt.cla()
        plt.hist(data[num],bins=20)
    
    fig=plt.figure()
    
    if frame_no==None:
        anim1 = animation.FuncAnimation(fig, update_quiver,frames=len(vectordata),fargs=(div_all,), interval=500, blit=True)
        plt.show()
    else:
        anim1=update_quiver(frame_no,div_all)
        plt.yscale('log')
    plt.xlabel('Divergence')
    plt.ylabel('Occurence')
    fig.tight_layout()
    return anim1   
    
#%%run to get data
x17= RunState(50,2000,64,64,5,1,0.17,5,20,1)
x22= RunState(50,2000,64,64,5,1,0.22,5,20,1)
x0 = RunState(50,2000,64,64,5,1,0.15,5,20,1)
x1 = RunState(50,2000,64,64,5,1,0.1,5,20,1)
x2 = RunState(50,2000,64,64,5,1,0.2,5,20,1)
x3 = RunState(50,2000,64,64,5,1,0.3,5,20,1)
#%%visualise microscopic data
MovieNodes(x22)
#writergif = animation.PillowWriter()
#MovieNodes(x2).save('RunState(50,2000,64,64,5,1,0.2,5,20,1).gif',writer=writergif)
#%%obtain the vectors for each node
Vecx17=Resultant_Vectors(x17[5],x17[6],x17[2],x17[1],x17[7],5,64,outv=True)
Vecx22=Resultant_Vectors(x22[5],x22[6],x22[2],x22[1],x22[7],5,64,outv=True)
Vecx0=Resultant_Vectors(x0[5],x0[6],x0[2],x0[1],x0[7],5,64,outv=True)
Vecx1=Resultant_Vectors(x1[5],x1[6],x1[2],x1[1],x1[7],5,64,outv=True)
Vecx2=Resultant_Vectors(x2[5],x2[6],x2[2],x2[1],x2[7],5,64,outv=True)
Vecx3=Resultant_Vectors(x3[5],x3[6],x3[2],x3[1],x3[7],5,64,outv=True)
#%%data for pixelated vectors
VectorPixels22=PixelatedVectors(x22,Vecx1,16,16)
VectorPixels17=PixelatedVectors(x17,Vecx1,16,16)
VectorPixels0=PixelatedVectors(x0,Vecx1,16,16)
VectorPixels1=PixelatedVectors(x1,Vecx1,16,16)
VectorPixels2=PixelatedVectors(x2,Vecx2,16,16)
VectorPixels3=PixelatedVectors(x3,Vecx3,16,16)
#%%movie for pixels
VectorMovie(VectorPixels2[0],VectorPixels2[1],None)
#%%for sinus rythm, data 1
divergence_histogram(VectorPixels0[0],VectorPixels0[1], log_scale = True, curve_fitting = True)
#divergence_map(VectorPixels1[0],VectorPixels1[1],5)
#HistoMovie(VectorPixels0[0],VectorPixels0[1],49)
#plt.savefig('Gauss0',dpi=1000)
#%%for sinus rythm, data 2
#divergence_histogram(VectorPixels2[0],VectorPixels2[1], log_scale = True, curve_fitting = True)
#divergence_map(VectorPixels2[0],VectorPixels2[1],1)
sr=AFIdentify(VectorPixels2[0],VectorPixels2[1],0.2)
print(sorted(sr , key=lambda k: [k[1], k[0]]))
VectorMovie(VectorPixels2[0],VectorPixels2[1],15)
#plt.savefig('Focal_e=0.2.png',dpi=1000)
#HistoMovie(VectorPixels2[0],VectorPixels2[1],49)
#plt.savefig('Div0.1Per.png',dpi=1000)
#%%
epsilon=[0.15,0.1,0.2,0.3,0.22,0.17]
a0=divergence_histogram(VectorPixels0[0],VectorPixels0[1], log_scale = True, curve_fitting = True)
a1=divergence_histogram(VectorPixels1[0],VectorPixels1[1], log_scale = True, curve_fitting = True)
a2=divergence_histogram(VectorPixels2[0],VectorPixels2[1], log_scale = True, curve_fitting = True)
a3=divergence_histogram(VectorPixels3[0],VectorPixels3[1], log_scale = True, curve_fitting = True)
a22=divergence_histogram(VectorPixels22[0],VectorPixels22[1], log_scale = True, curve_fitting = True)
a17=divergence_histogram(VectorPixels17[0],VectorPixels17[1], log_scale = True, curve_fitting = True)
alpha=[a0[0],a1[0],a2[0],a3[0],a22[0],a17[0]]
alphaer=[a0[1],a1[1],a2[1],a3[1],a22[1],a17[1]]
#%%plotting alpha against epsilon
plt.plot(epsilon,alpha,'bx')
plt.errorbar(epsilon,alpha,yerr=alphaer,fmt='o',mew=2, ms=3)
plt.xlabel(r'Probability of cell not exciting,$\epsilon$')
plt.ylabel(r'$\alpha$, coefficient of Gaussian fit')
plt.savefig('AFindicator',dpi=1000)
#%%test dot product
rd=ReentrantDot([VectorPixels2[0][17]],VectorPixels2[1],16,16)
rd[0]=[1000 if i==None else abs(i) for i in rd[0]]
minl=rd[0].index(min(rd[0]))
loc=[minl%16,minl//16]
print(loc)
#%%
divergence_histogram(VectorPixels0[0],VectorPixels0[1], log_scale = True, curve_fitting = True)
divergence_histogram(VectorPixels2[0],VectorPixels2[1], log_scale = True, curve_fitting = True)
plt.savefig('2histtog',dpi=1000)