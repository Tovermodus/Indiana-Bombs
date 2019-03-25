
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
from agent_code.indiana_bombs.q_funct import Qf
from settings import s
from settings import events as eventnames
from datetime import datetime

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.Q = Qf()
    self.Q.setup()
    self.Q.load()
    self.historyn = np.array([np.zeros(182)])
    self.last_action = 0
    self.coins = 0
    self.crates = 0
    self.kills = 0
    self.steps = 0
    millis = datetime.now().microsecond
    np.random.seed(millis)
    
def get_blast_coords(x,y, arena):
        blast_coords = [(x,y)]

        for i in range(1, s.bomb_power+1):
            if( x+i < 15):
                if arena[x+i,y] == 1: break
                blast_coords.append((x+i,y))
        for i in range(1, s.bomb_power+1):
            if(x-i >=0):
                if arena[x-i,y] == 1: break
                blast_coords.append((x-i,y))
        for i in range(1, s.bomb_power+1):
            if(y+i < 15):
                if arena[x,y+i] == 1: break
                blast_coords.append((x,y+i))
        for i in range(1, s.bomb_power+1):
            if(y-i >=0):
                if arena[x,y-i] == 1: break
                blast_coords.append((x,y-i))

        return blast_coords

def arenatosavedata(value):
    return ((value + 0.5)%3-1.5)
def arenaval(x,y, arena, sixteen=False):
    if(sixteen):
        if(x < 0 or x >= 17 or y < 0 or y >= 17):
            return 1
        else:
            return arena[x,y]
    if(x < 0 or x >= 15 or y < 0 or y >= 15):
        return 1
    else:
        return arena[x,y]
def coboval(x,y,coins,bombs, explosions, others, arena):
    if (0<= x and 0 <= y and x<15 and y < 15):
        if(explosions[x,y] != 0):
            return -7
    for i in range(len(bombs)):
        b_coords = get_blast_coords(bombs[i][0]-1, bombs[i][1]-1, arena)
        for coord in b_coords:
            if(coord[0] == x and coord[1] == y):
                return -5+bombs[i][2]
    
    for i in range(len(coins)):
        if(coins[i][0]-1 == x and coins[i][1] - 1 == y):
            return 10
        if(abs(coins[i][0]-1-x) + abs(coins[i][1]-1-y) == 1):
            return 8
        if(abs(coins[i][0]-1-x) + abs(coins[i][1]-1-y) == 2):
            return 6
    for i in range(len(others)):
        if(others[i][0]-1 == x and others[i][1] - 1 == y):
            return -10
    return 0.1
def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """
    calc_state(self)
    T = 1
    self.vorletzte = self.last_action
    nextActionData = np.array([np.append(j,self.sd) for j in range(6)])
    nDeval = self.Q.predict(nextActionData)
    enDeval = np.exp(nDeval/T)
    bsum = np.sum(enDeval)
    boltz = enDeval / bsum
    if(np.random.random_sample() < 0.0 ):
        rand = np.random.random_sample()
        if(rand < np.sum(boltz)):
            new_action = 5
        if(rand < np.sum(boltz[:5])):
            new_action = 4
        if(rand < np.sum(boltz[:4])):
            new_action = 3
        if(rand < np.sum(boltz[:3])):
            new_action = 2
        if(rand < np.sum(boltz[:2])):
            new_action = 1
        if(rand < boltz[0]):
            new_action = 0
    else:
        new_action = np.argmax(nDeval)    
 
    if(new_action == 4): 
        self.next_action = 'WAIT'
    if(new_action == 1):
        self.next_action = 'UP'
    if(new_action == 2):
        self.next_action = 'DOWN'
    if(new_action == 3):
        self.next_action = 'LEFT'
    if(new_action == 0):
        self.next_action = 'RIGHT'
    if(new_action == 5):
        self.next_action = 'BOMB'
    self.last_action = new_action
    
            
   
   
def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    ret = -2
    self.steps = self.steps + 1
    for ev in self.events:
        if(ev <=3):#mov
            ret += 3
        if(ev == 6):#inv
            ret += -3
            if(self.bomb =='hot'):
                ret += -3
        if(ev == 7):#bom
            ret += 1
        if(ev == 9):#cra
            self.historyn[-5,0] += 30
            self.crates = self.crates + 1
        if(ev == 11):#coi
            self.historyn[-5:,0] += 30
            self.coins = self.coins + 1
            ret += 100
        if(ev == 12):#kil
            self.historyn[-5:,0] += 500
            ret += 100
            self.kills = self.kills + 1
        if(ev == 13):#sui
            self.historyn[-5, 0] += -300
            self.historyn[-4:,0] += -100
            ret +=-50
        if(ev == 14):#dea
            ret += -200
            self.historyn[-4:,0] += -100
        if(ev == 16):#sur
            self.steps = 1000
    if(self.vorletzte == 0 and self.last_action == 3):
        ret -= 0.2
    if(self.vorletzte == 3 and self.last_action == 0):
        ret -= 0.2
    if(self.vorletzte == 2 and self.last_action == 2):
        ret -= 0.2
    if(self.vorletzte == 1 and self.last_action == 1):
        ret -= 0.2
    if(self.bomb == 'hot'):
        ret -= -3
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    self.historyn = np.append(self.historyn,[np.concatenate(([ret],[self.last_action],self.sd))], axis=0)

    
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    calc_state(self)
    self.historyn =  np.append(self.historyn,[np.concatenate(([0],[-1],self.sd))], axis=0)
    history = np.load("training_data.npy")
    history = np.append(history, self.historyn[1:], axis=0)
    #history = self.historyn[1:]
    np.save("training_data", history[-300000:])
    score = np.concatenate(([self.coins], [self.kills], [self.crates], [self.steps]))
    scores = np.load("scores.npy")
    scores = np.append(scores, [score], axis = 0)
    np.save("scores.npy", scores)
    self.coins = 0
    self.crates = 0
    self.kills = 0
    self.steps = 0
    self.historyn = np.array([np.zeros(182)])


def calc_state(self):
    arena = arenatosavedata(self.game_state['arena'][1:-1,1:-1])
    
    x = self.game_state['self'][0] - 1
    y = self.game_state['self'][1] - 1
    check_vals = [[x,y],[x-1,y],[x-2,y],[x-3,y],[x-4,y],[x-5,y],[x-6,y],[x+1,y],[x+2,y],[x+3,y],[x+4,y],[x+5,y],[x+6,y],
                  [x,y+1],[x-1,y+1],[x-2,y+1],[x-3,y+1],[x-4,y+1],[x-5,y+1],[x+1,y+1],[x+2,y+1],[x+3,y+1],[x+4,y+1],[x+5,y+1],
                  [x,y+2],[x-1,y+2],[x-2,y+2],[x-3,y+2],[x-4,y+2],[x+1,y+2],[x+2,y+2],[x+3,y+2],[x+4,y+2],
                  [x,y+3],[x-1,y+3],[x-2,y+3],[x-3,y+3],[x+1,y+3],[x+2,y+3],[x+3,y+3],
                  [x,y+4],[x-1,y+4],[x-2,y+4],[x+1,y+4],[x+2,y+4],[x,y+5],[x-1,y+5],[x+1,y+5],[x,y+6],
                  [x,y-1],[x-1,y-1],[x-2,y-1],[x-3,y-1],[x-4,y-1],[x+1,y-1],[x+2,y-1],[x+3,y-1],[x+4,y-1],[x+5,y-1],[x-5,y-1],
                  [x,y-2],[x-1,y-2],[x-2,y-2],[x-3,y-2],[x+1,y-2],[x+2,y-2],[x+3,y-2],[x-4,y-2],[x+4,y-2],
                  [x,y-3],[x-1,y-3],[x-2,y-3],[x+1,y-3],[x+2,y-3],[x+3,y-3],[x-3,y-3],
                  [x,y-4],[x-1,y-4],[x+1,y-4],[x,y-5],[x+2,y-4],[x-2,y-4],[x-1,y-5],[x+1,y-5],[x,y-6]]
 
    save_data = np.zeros(len(check_vals)*2+10)
    save_data[0] = x/15
    save_data[1] = y/15
    closest_enemy = [-100,100]
    dist = 20
    for o in self.game_state['others']:
        if(abs(o[0] - 1 - x) + abs(o[1] - 1 - y) < dist):
            closest_enemy[0] = o[0] - 1
            closest_enemy[1] = o[1] - 1
            dist = abs(o[0] - 1 - x) + abs(o[1] - 1 - y)
    save_data[2] = dist/20
    save_data[3] = (closest_enemy[0] - x)/15
    save_data[4] = (closest_enemy[1] - y)/15
    closest_coin = [100,-100]
    dist = 20
    box_dists = np.transpose(np.array(np.where(arena == 0))) - np.array([x,y])
    if(box_dists.size > 0):
        min_dist_box = np.argmin(np.sum(np.absolute(box_dists), axis = 1))
        save_data[8] = box_dists[min_dist_box][0]/15
        save_data[9] = box_dists[min_dist_box][1]/15
    
    for o in self.game_state['coins']:
        if(abs(o[0] - 1 - x) + abs(o[1] - 1 - y) < dist):
            closest_coin[0] = o[0] - 1
            closest_coin[1] = o[1] - 1
            dist = abs(o[0] - 1 - x) + abs(o[1] - 1 - y)
    self.dist = min(10,dist/2)
    save_data[5] = dist/20
    save_data[6] = (closest_coin[0] - x)/15
    save_data[7] = (closest_coin[1] - y)/15
    
    for i in range(len(check_vals)):
        save_data[8+i] = arenaval(check_vals[i][0],check_vals[i][1],arena)
    for i in range(len(check_vals)):
        save_data[8+len(check_vals) + i] = coboval(check_vals[i][0],check_vals[i][1],self.game_state['coins'], self.game_state['bombs'], self.game_state['explosions'][1:-1,1:-1], self.game_state['others'], arena)/10
    self.sd = save_data
    if(coboval(x,y,self.game_state['coins'], self.game_state['bombs'], self.game_state['explosions'][1:-1,1:-1], self.game_state['others'],arena) <= 0):
        self.bomb = 'hot'
    else:
        self.bomb = 'cold'
