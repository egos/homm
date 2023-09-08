import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
import time
import itertools 
from types import SimpleNamespace
import networkx as nx
from networkx.algorithms import approximation as approx
from utils import *
# import cv2

st.set_page_config(page_title="sic_/",
    page_icon="🎯",
    layout="wide",
)

def Draw(algo): 
    pas = algo.pas
    img  = Image.fromarray(algo.arr.astype('uint8'), 'RGB')
    # font = ImageFont.truetype("arial.ttf", 15)
    draw = ImageDraw.Draw(img)
    pos  = tuple(zip(*np.where(algo.env.grid >= 1)))
   
    G = algo.G
    if G is not None:              
        for i, (n1,n2, data) in enumerate(G.edges(data = True)):
            # print(n1,n2, i)
            L = data['path']
            L = np.array(L)*algo.pas + algo.pas/2
            L = list(map(tuple, L))
            for j in range(len(L)-1):
                shape = [L[j], L[j+1]]
                draw.line(shape, fill =(255, 255, 255), width = 3)

    i = 0
    grid = algo.env.grid
    for point in pos:
        i = algo.gridC[point]
        x,y = np.array(point)*pas
        if grid[point] == 1:
            draw.rectangle((x,y, x+pas,y+pas), fill=(0, 0, 0), outline=(255, 255, 255))
        if grid[point] == 2:
            draw.rectangle((x,y, x+pas,y+pas), fill=(0, 192, 192), outline=(255, 255, 255))            
            text = 'R' + str(i)
            if i == 0 : text = 'R'
            draw.text((x+pas/2,y + pas/2), text, align ="center",anchor="mm")
            
        if grid[point] == 3:
            draw.rectangle((x,y, x+pas,y+pas), fill=(150, 150, 0), outline=(255, 255, 255))
            draw.text((x+pas/2,y + pas/2), 'H', align ="center",anchor="mm")
   
    return img

def Draw_CV2(algo): 
    pas = algo.pas
    arr = algo.arr.copy()
    pos  = tuple(zip(*np.where(algo.env.grid >= 1)))
    
    G = algo.G
    if G is not None:
        for n1,n2, data in G.edges(data = True):
            L = data['path']
            L = (np.array(L)*algo.pas + algo.pas/2).astype(int)
            L = list(map(tuple, L))
            for j in range(len(L)-1):
                shape = [L[j], L[j+1]]
                # print(shape)
                # arr = cv2.arrowedLine(arr, L[j], L[j+1], (255,255,255), 2,tipLength = 0.2)
                arr = cv2.line(arr, L[j], L[j+1], (255,255,255), 2)

                # draw.line(shape, fill ="white", width = 4)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # i = 0
    # for point in pos:
    #     x,y = np.array(point)*pas
    #     CENTER = (x+pas//2, y+pas//2)
    #     TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    #     TEXT_SCALE = 0.7
    #     TEXT_THICKNESS = 1
    #     TEXT = "0"
    #     text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    #     # text_origin = tuple(np.array((CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)).astype(int))
    #     text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)
    #     if algo.A[point] == 1:
    #         arr = cv2.rectangle(arr, (x,y), (x+pas,y+pas), (0, 0, 0), -1)            
    #     if algo.A[point] == 2:
    #         arr = cv2.rectangle(arr, (x,y), (x+pas,y+pas), (0, 192, 192), -1)
    #         text = 'R' + str(i)
    #         arr = cv2.putText(arr, text, text_origin, TEXT_FACE,TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)
    #         i+=1
    #     if algo.A[point] == 3:
    #         arr = cv2.rectangle(arr, (x,y), (x+pas,y+pas), (150, 150, 0), -1)
    #         arr = cv2.putText(arr, 'H', text_origin, TEXT_FACE,TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)
    #     arr = cv2.rectangle(arr, (x,y), (x+pas,y+pas), (255, 255, 255), 1)

    img  = Image.fromarray(arr.astype('uint8'),'RGB')
    fnt  = ImageFont.truetype('/root/pyscripts/arial.ttf',20)
    draw = ImageDraw.Draw(img)
    i = 0
    for point in pos:
        x,y = np.array(point)*pas
        if algo.A[point] == 1:
            draw.rectangle((x,y, x+pas,y+pas), fill=(0, 0, 0), outline=(255, 255, 255))
        if algo.A[point] == 2:
            draw.rectangle((x,y, x+pas,y+pas), fill=(0, 192, 192), outline=(255, 255, 255))
            # print(y,x)
            text = 'R' + str(i)
            draw.text((x+pas/2,y + pas/2), text,font=fnt, align ="center",anchor="mm")
            i+=1
        if algo.A[point] == 3:
            draw.rectangle((x,y, x+pas,y+pas), fill=(150, 150, 0), outline=(255, 255, 255))
            draw.text((x+pas/2,y + pas/2), 'H',font=fnt, align ="center",anchor="mm")
   
    return img

def reset_algo(size, custom = False):

    pos = [(5,5),(1,3),(9,3), (9,9)] 
    env = env_([], custom, size)
    size = env.size
    grid0 = env.grid0
    grid = env.grid   
    pas = 800//size
    arr0 = np.full((size*pas, size*pas,3), 0)
    arr0[:,:,:] = [150,150,150]
    arr0[::pas,:,:] = 200
    arr0[-1,:,:] = 200
    arr0[:,::pas,:] = 200
    arr0[:,-1,:] = 200
    
    # print(pas)
    algo = dict(
        pas = 800//size,
        size = size, 
        points = [],
        arr0 = arr0,
        arr = arr0.copy(),
        # grid = grid,
        # grid0 = grid0,
        paths = [],
        count = 0,
        env = env,
        G = None,
        cycle = [],
        gridC = [],
        )
    algo = SimpleNamespace(**algo)

    return algo

size = 10
pas = 30
c1, c2 = st.columns([0.3, 0.7])

# pas = c1.selectbox('pas', [10,20,30,40],3)
custom = c1.toggle('custom map')
if not custom :
    size = c1.selectbox('size', [10,20,30,40])

if "algo" not in st.session_state:
    st.session_state["algo"] = reset_algo(size, custom)
    st.session_state["value"]  = None

if c1.button('reset'): 
    # st.session_state["points"] = []
    st.session_state["algo"] = reset_algo(size, custom)
    st.experimental_rerun()

algo   = st.session_state["algo"]
pas    = algo.pas

modetext = 'map change : 1: wall / 2: R / 3: H'
mode = c1.slider(modetext,1,3,step = 1)
Graph = c1.toggle('process path')
algo.env.PathMode = c1.checkbox('diamond step')*1
greedy_tsp = c1.checkbox('greedy_tsp')

grid = algo.env.grid
# print(grid)
algo.cycle = []
algo.G = None
algo.gridC = np.zeros(algo.env.grid.shape, dtype=int)
if Graph:
    tic = time.perf_counter()  
    G = Grid_Path(algo.env)     
    if greedy_tsp and G:        
        cycle = approx.greedy_tsp(G, source="H", weight= 'dist')
        for i, n in enumerate(cycle):
            #  print(i,n, G.nodes[n])
             algo.gridC[G.nodes[n]['pos']] = i            
            
        cycle = nx.utils.pairwise(cycle, cyclic=False)    
        edges = list(cycle)[:-1]
        algo.cycle = edges
        G = G.edge_subgraph(edges)

    toc = time.perf_counter()
    dt = round(toc - tic , 6)
    c1.write("{} / runtime : {}".format(G, round(toc - tic , 6)))
    print(G, dt)
    algo.G = G

img = Draw(algo)     
value = None


# runActif = (len(algo.cycle)>0)
if c1.toggle('run') & (len(algo.cycle)>0) & greedy_tsp:
    gridC = algo.gridC
    grid = algo.env.grid
    pos = np.where(grid == 3)
    grid[pos] = 0
    pos = np.where(gridC == 1)
    grid[pos] = 3
    with c2 :
        value = streamlit_image_coordinates(img, key="pil")
    st.session_state["value"] = value
    st.session_state["algo"] = algo
    time.sleep(0.3)
    st.experimental_rerun()
# if c1.button('run') & (len(algo.cycle)>0):
#     for i,n in range(algo.cycle)
#     algo.G = None
#     gridRun = grid.copy()
#     slot = c2.empty()
#     pos = tuple(zip(*np.where(grid >= 2)))
#     img = Draw(algo)
#     for i in range(len(pos)): 
#         pos = tuple(zip(*np.where(algo.A >= 2)))
#         idx = np.random.choice(len(pos),1)[0] 
#         algo.A[pos[idx]] = 0       
#         img = Draw(algo)
#         slot.image(img)
#         # value = streamlit_image_coordinates(img, key="pil")
#         time.sleep(0.1)
#     st.experimental_rerun()
else :  
    with c2 :
        value = streamlit_image_coordinates(img, key="pil")

        if (value is not None):
            if value!= st.session_state["value"] :
                algo.count+=1
                point = int(value["x"]), int(value["y"])
                y,x = np.array(point)//pas
                grid = algo.env.grid
                if mode == 3:
                    pos = np.where(grid == mode)
                    grid[pos] = 0
                    grid[y,x] = mode
                else : 
                    if grid[y,x] >= 1:
                        grid[y,x] = 0
                    elif grid[y,x] == 0: 
                        grid[y,x] = mode

                st.session_state["value"] = value
                st.session_state["algo"] = algo
                st.experimental_rerun()



