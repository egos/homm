import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import io
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import skimage
import skimage.transform

print('begin')

def Path1(A,start): 
    N = len(A)
    v0 = np.array([-1,1,-N,N])
    v0 = np.array([-1,1,-N,N, -N-1, -N+1,N+1,N-1])
    Dim = len(v0)
    e = 2
    A[start] = e
    a = A.reshape(-1)
    v = np.where(a == e)  

    while len(v) > 0 :
        v = np.tile(v, (Dim, 1)).T + v0
        v = v[np.where(a[v]==0)]
        v = np.unique(v)        
        e+=1
        a[v]=e
    return a.reshape((N,N))

def Path2(A,start,goal):
    N = len(A)
    v0 = np.array([-1,1,-N,N])
    v0 = np.array([-1,1,-N,N, -N-1, -N+1,N+1,N-1])
    Dim = len(v0)
    e1,e2  = A[start] , A[goal]
    a = A.reshape(-1)
    v = goal[1] + goal[0]* N
    L  = [goal]
    while e2 > 2:
        v = v + v0
        v = v[np.where(a[v] == e2-1)][0]
        a[v] = e2
        e2 -= 1
        pos = (int(np.ceil(v/N)-1),  v%N)
        L.insert(0,pos)
    return L

class heroes():
    def __init__(self, pos, move_point):
        self.move_point_ini  = move_point
        self.move_point = 0
        self.pos = pos
        self.money = 0
        self.update()       
        
    def update(self):
        self.move_point = self.move_point_ini

def board(Sa,Spos, frame): 
    A = Sa[frame]
    fig, ax = plt.subplots()
    f = ax.imshow(A)
    x, y = Spos[frame]['pos']
    f = ax.scatter(y, x, s=70, c='red', marker='x')   
    return fig

@st.cache(allow_output_mutation=True)
def run(Tpos_ini, epoch):
    print('run')

    A0 = np.genfromtxt('A.csv', delimiter=",", dtype=int).T
    N = A0.shape[0]
    A = A0.copy()
    T = np.zeros(A.shape)
    # Tpos_ini = [(5,5),(1,3), [7,7]]
    pos = tuple(zip(*Tpos_ini))
    T[pos] = 5   
    start = (3,6)
    H1 = heroes(start,2)

    Sa = [T+A]
    Spos = [vars(H1).copy()]
    
    for i in range(epoch):
        A1 = Path1(A.copy(),H1.pos)
        
        #IA
        Tpos = np.where(T == 5)
        LenList = A1[Tpos]-2
        idx = np.argmin(LenList)
        Len = LenList[idx]
        goal = tuple(zip(*Tpos))[idx]

        #ACTION
        L = Path2(A1.copy() ,start,  goal)
        # L,idx, Len
        if H1.move_point > Len :
            move = Len
        else :
            move = H1.move_point
            
        #RESULT
        NewPos = L[move]
        H1.pos = NewPos
        H1.move_point -= move
        if T[NewPos] == 5 : 
            T[NewPos] = 0
            H1.money += 1
        Sa.append(T+A)
        Spos.append(vars(H1).copy())
            
    #     vars(H1)
        H1.update()
    return Sa, Spos

def Grid_update(Sa, Spos, EPOCH, FRAME):
    arr = Sa[FRAME]
    arr = arr/10
    PAS = 20
    arr = skimage.transform.resize(arr, (16*PAS, 16*PAS), order =0)
    x,y= Spos[FRAME]['pos']
    arr[int(PAS*(x + 0.4)):int(PAS*(x+0.6)),int(PAS*(y + 0.4)):int(PAS*(y+0.6))] = 0.9
    return arr
    

pos = [(5,5),(1,3),(9,3), [9,9]] 
EPOCH = 10
Sa, Spos= run(pos, EPOCH) 


animate = st.sidebar.button('animate')
slider_ph = st.empty()
H1 = st.sidebar.empty()
FRAME = slider_ph.slider("frame", 0,EPOCH, key = '0')  
arr = Grid_update(Sa, Spos, EPOCH, FRAME)
H1.write(Spos[FRAME]) 
IMG = st.image(arr, width =500)
st.sidebar.image(Grid_update(Sa, Spos, EPOCH, 0), clamp = False)


if animate:
    print("animate")
    for FRAME in range(EPOCH):
        print(Spos[FRAME])
        # LAP = slider_ph.slider("slider", 0, EPOCH, key = x)
        arr = Grid_update(Sa, Spos, EPOCH, FRAME)
        IMG.image(arr,width =500, output_format = 'PNG')
        H1.write(Spos[FRAME])
        time.sleep(.05)  
