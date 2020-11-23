import numpy as np
import numpy.random as nr
import time
import copy

import pickle

import math

import matplotlib.pyplot as plt

def step_model(state, a):
    #Take a state and an action, return the next state, reward, and whether
    #the terminal state was entered
    
    row= state[0]
    col= state[1]
    
    if(nr.rand() >= 0.70):
        a= nr.choice(np.delete(A,a))
    
    if a == 0:
        next_state= (row + 1, col)
    if a == 1:
        next_state= (row - 1, col)
    if a == 2:
        next_state= (row, col + 1)
    if a == 3:
        next_state= (row, col - 1)
    
    
    if(r_type_norm):
        reward= 0
    else:
        reward = -1
    
    if(next_state in terminal_list):
        if(r_type_norm):
            reward+= rewards_list_norm[terminal_list.index(next_state)]
        else:
            reward+= rewards_list[terminal_list.index(next_state)]

        return list(next_state), reward, 1
    
    if(next_state in obstacle_list):
        next_state= state
        
        if(not r_type_norm):
            reward += -5
            
        return list(next_state),reward,0

    if(0 <= next_state[0] <= grid_size-1 and  0 <= next_state[1] <= grid_size-1 ):
        return list(next_state),reward,0

    else:
        next_state = state
        return list(next_state),reward,0
    
def QL_episode(Q, state= None, greedy= False):
    #Perform Q learning episode (pg. 131)
    #e_greedy input: 1 if using epsilon-greedy. 0 if using random uniform policy
    if(state is None):
        #state= list(nr.randint(0, high= grid_size, size= (1,4))[0])
        state= [0,0]
    
    G=0
    for x in range(3000):
        if greedy:
            a= np.argmax(Q[tuple(state)])
        else:
            a= epsilon_greedy(state,Q)
            # a= softmax(state,Q)
        
        state_act= tuple(state + [a])
        next_state, r, terminate = step_model(tuple(state), a)
        
        G += r
        if terminate:
            Q[state_act]= Q[state_act] + alpha*(r - Q[state_act])
            break
        else:
            Q[state_act]= Q[state_act] + alpha*(r + gamma*np.max(Q[tuple(next_state)]) - Q[state_act])
            state= next_state
            
    return Q,G

def epsilon_greedy(state,Q):
    #Given a state space, action space, and Q matrix, perform epsilon greedy to choose next action. 
    #Ties between the best actions are broken randomly.
    
    best_action= nr.choice(np.flatnonzero(Q[tuple(state)] == Q[tuple(state)].max()))
    if(nr.rand() < 1 - epsilon ):
        a= best_action
    else:
        a= nr.choice(np.delete(A,best_action))
    
    return a

def softmax(state,Q):
    act_vals= Q[tuple(state)]
    num= np.exp(act_vals / temp)
    act_dist= num / np.sum(num)
    return nr.choice(A, p= act_dist)
    

def teach_model(N_episodes):
    #Teach the model on N_episodes.
    #MC = 1 Use Monte-Carlo, MC = 0 Use Q-Learning
    #e_greedy input: 1 if using epsilon-greedy. 0 if using random uniform policy
    
    Q= np.zeros([grid_size, grid_size, 4])
    G_arr= []
    try:
        for n in range(N_episodes):
            if(n > 2900):
                Q,G= QL_episode(Q, greedy= True)
            else:
                Q,G= QL_episode(Q, greedy= False)
            G_arr.append(G)
    finally:
        return sum(G_arr[-100:])/100

def gen_obstacles():
    n_obstacles= int(math.floor(0.1*(grid_size*grid_size-3)))
    taken_list= copy.copy(terminal_list)
    taken_list.append((0,0))
    
    obstacle_list= []
    while len(obstacle_list) != n_obstacles:
        ob= tuple(nr.randint(0, high= grid_size, size= (1,2))[0])
        if(ob not in taken_list):
            obstacle_list.append(ob)
        else:
            pass
    return obstacle_list

def validity_test():
    goal_check_list= [False, False, False]
    
    state= [0,0]
    for x in range(3000):
        a= nr.choice(A)
        state, reward, terminate = step_model(state,a)
        if( tuple(state) in terminal_list):
            goal_check_list[terminal_list.index(tuple(state))] = True
            
        if(all( check == True for check in goal_check_list)):
            return True
        
        if(terminate):
            state= [0,0]
    return False
        

#Parameters defined globally
    
all_obstacles= pickle.load(open('C:\\Users\\joeji\\Documents\\valid_courses.p','rb'))

grid_size= 10
start_state= (0,0)

terminal_list=[(0,9),(9,0),(7,7)]

rewards_list= [50,50,200]
rewards_list_norm=[0.5,0.5,1]

A= np.array([0,1,2,3])

r_type_norm= False
epsilon=0.36
alpha= 0.83
gamma= 0.98
temp= 8.79

if __name__ == '__main__':
    Q_init= np.zeros([grid_size, grid_size, 4])
    #Q_init= np.ones([grid_size, grid_size, 4])
    
    R_arr=[]
    for obstacle_list in all_obstacles[0:10]:
        # obstacle_list=[(3,0),(4,0),(8,0),(9,2),(6,4),(7,5),(3,9),(4,8),(9,8)]
        R= teach_model(3000)
        R_arr.append(R)
        print(R)
        
    print(sum(R_arr)/10)
        