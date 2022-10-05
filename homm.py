"""HOMM implemented with pyxel.

Controls are the arrow keys ← ↑ → ↓

Q: Quit the game
R: Restart the game

Created by bertrand pelletier  in 2022.

nuitka --standalone --onefile --windows-disable-console --include-data-dir=assets=assets --windows-icon-from-ico=assets/homm.ico --follow-imports homm.py
"""

from collections import deque, namedtuple
import numpy as np
import pyxel


Point = namedtuple("Point", ["x", "y"])  # Convenience class for coordinates

COL_BACKGROUND = 3
COL_BODY = 11
COL_HEAD = 7
COL_DEATH = 8
COL_APPLE = 8

TEXT_DEATH = ["GAME OVER", "(Q)UIT", "(R)ESTART"]
COL_TEXT_DEATH = 0
HEIGHT_DEATH = 5

UI = 64
WIDTH = 128+UI
HEIGHT = 128
SIZE = 128
PAS = 8
NMAX = int((SIZE)/PAS)


HEIGHT_SCORE = pyxel.FONT_HEIGHT
COL_SCORE = 6
COL_SCORE_BACKGROUND = 5

UP = (0, -1)
DOWN = (0, 1)
RIGHT = (1, 0)
LEFT = (-1, 0)

move = False

START = (1,1)

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

def check_mouse(m,A):
    N = len(A)
    cond  = (m[0] < N) & (m[0] > 0) & (m[1] < N) & (m[1] > 0)
    if cond:
        if A[m] == 1:
            cond = False  
    print(cond)      
    return cond

class Snake:
    """The class that sets up and runs the game."""
    test = 'b'

    def __init__(self):
        """Initiate pyxel, set up initial game variables, and run."""
        
        pyxel.init(WIDTH, HEIGHT, title="HOMM", fps=20,capture_scale=1)#, display_scale=15, capture_scale=5
        pyxel.load("assets/rpg01.pyxres")
        self.tm = pyxel.tilemap(0)
        self.img = pyxel.image(0)
        
        print(self.tm.pget(2,2))
        
        pyxel.mouse(True)
        self.reset()       
        pyxel.run(self.update, self.draw)
              
    def reset(self):
        """Initiate key variables (direction, snake, apple, score, etc.)"""
        self.goal = ()
        self.path = []
        self.direction = RIGHT
        self.pos = START        
        self.move = False 
        self.click = False  
        self.step = 0           
        A = np.zeros((NMAX,NMAX))
        B = np.zeros((NMAX,NMAX))
        for i in range(NMAX):
            for j in range(NMAX):
                tile = self.tm.pget(i,j)
                B[i,j] = tile[0] + tile[1]*10
                if self.tm.pget(i,j) in [(4,5),(5,5),(6,5)]:
                    A[i,j] = 1
                    
        np.savetxt('A.csv', A, delimiter=',', fmt='%f')
        np.savetxt('B.csv', B, delimiter=',', fmt='%f')
        #A[:,0], A[:,NMAX], A[0,:],A[NMAX,:] = 1,1,1,1
        self.A = A

    def update(self):
        """Update game"""
        self.update_direction()
        self.mouse = tuple(np.ceil([pyxel.mouse_x/PAS-1, pyxel.mouse_y/PAS-1]).astype(int))
        if pyxel.btn(pyxel.KEY_Q):   pyxel.quit()
        if pyxel.btnp(pyxel.KEY_R):  self.reset()
        if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT) :
            if check_mouse(self.mouse,self.A):
                print(self.pos, self.mouse)
                if self.mouse == self.goal:
                    self.click = True
                else :
                    self.click = False
                    self.step = 0
                    self.goal = self.mouse            
                    goal = tuple(self.goal)
                    A1   = Path1(self.A.copy(),self.pos)
                    self.path = Path2(A1.copy() ,self.pos,  self.mouse)
            else : 
                self.path = []
                self.goal = self.pos
        if self.click :
            if self.step < len(self.path):        
                self.pos = self.path[self.step]
                self.step+=1
            else :
                print('end')
                self.click = False
                self.step = 0
                self.goal = self.pos
                self.path = []
                                  
    def update_direction(self):
        """Watch the keys and change direction."""

        if pyxel.btn(pyxel.KEY_UP) or pyxel.btn(pyxel.GAMEPAD1_BUTTON_DPAD_UP):
            self.direction = UP
            self.move = True
        elif pyxel.btn(pyxel.KEY_DOWN) or pyxel.btn(pyxel.GAMEPAD1_BUTTON_DPAD_DOWN):
            self.direction = DOWN
            self.move = True
        elif pyxel.btn(pyxel.KEY_LEFT) or pyxel.btn(pyxel.GAMEPAD1_BUTTON_DPAD_LEFT):
            self.direction = LEFT
            self.move = True
        elif pyxel.btn(pyxel.KEY_RIGHT) or pyxel.btn(pyxel.GAMEPAD1_BUTTON_DPAD_RIGHT):
            self.direction = RIGHT
            self.move = True
        new_pos = (self.pos[0] + self.direction[0], self.pos[0] + self.direction[1])
        if (self.A[new_pos] != 1) & (self.move):
            self.pos= (new_pos[0], new_pos[1])
            self.move = False            

    def draw(self):
        """Draw the background, snake, score, and apple OR the end screen."""        
        
        pyxel.cls(col=0)
        
        tm = pyxel.tilemap(0)
        #pyxel.camera(self.pos[0], self.pos[1])
        pyxel.bltm(0, 0, self.tm, 0, 0, PAS*(NMAX), 8*(NMAX))

        if len(self.path)> 0: 
            gx, gy = self.goal
            pyxel.rectb(gx*PAS, gy*PAS, PAS, PAS, 8)
            for i,j in self.path[self.step:]: 
                pyxel.circ(i*PAS+PAS/2, j*PAS+PAS/2, 1, 8)
        pyxel.blt(self.pos[0]*PAS, self.pos[1]*PAS, self.img, 32, 0, PAS, PAS)
        
        
        """UI"""
        #score = f"{self.score:04}"
        #pyxel.camera(0, 0)
        pyxel.rectb(self.mouse[0]*PAS, self.mouse[1]*PAS, PAS, PAS, 7)
        pyxel.rect(SIZE, 0, WIDTH, HEIGHT, 4)
        pyxel.text(SIZE+1, 1, 'Mouse :' + str(self.mouse), COL_SCORE)
        pyxel.text(SIZE+1, 10,str((pyxel.mouse_x, pyxel.mouse_y)), COL_SCORE)
        pyxel.text(SIZE+1, 20,str(self.tm.pget(self.mouse[0],self.mouse[1])), COL_SCORE)
        
        #img = pyxel.image(0).set(0, 0, ["0123", "4567", "89ab", "cdef"])
        #print(img)
        #pyxel.image(img)
        # print(self.mouse)
        
    def draw_death(self):
        """Draw a blank screen with some text."""

        pyxel.cls(col=COL_DEATH)
        display_text = TEXT_DEATH[:]
        display_text.insert(1, f"{self.score:04}")
        for i, text in enumerate(display_text):
            y_offset = (pyxel.FONT_HEIGHT + 2) * i
            text_x = self.center_text(text, WIDTH)
            pyxel.text(text_x, HEIGHT_DEATH + y_offset, text, COL_TEXT_DEATH)

    @staticmethod
    def center_text(text, page_width, char_width=pyxel.FONT_WIDTH):
        """Helper function for calculating the start x value for centered text."""

        text_width = len(text) * char_width
        return (page_width - text_width) // 2


Snake()
