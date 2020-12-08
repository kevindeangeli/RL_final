import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
import numpy.random as nr
import argparse
import time
import sys
import pickle
import pandas as pd

import copy

import seaborn as sns

def drawWorld(map_size=5, agent_loc=(0,0), obstacle_loc_lst=[(3,3)],optimal_exit=(1,1),maze_exits_suboptimal=[(4,4)], pause = 2):
    '''
    :param map_size: map will be nXn squares
    :param agent_loc: a pair of coordinates (x,y)
    :param obstacle_loc_lst: list of tuples [(x1,y1),...,(x2,y2)]
    :param pause in seconds
    :maze_exits_suboptimal: list suboptimal exits
    :optimal_exit: tuple (x,y)
    :return: plotted map.
    '''
    wid = 1
    hei = 1
    nrows = map_size
    ncols = map_size
    inbetween = 0.1

    xx = np.arange(0, ncols, (wid+inbetween))
    yy = np.arange(0, nrows, (hei+inbetween))

    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')

    pat = []
    for idx1, xi in enumerate(xx):
        for idx2,yi in enumerate(yy):
            if (idx1,idx2) == agent_loc:
                sq = patches.Rectangle((xi, yi), wid, hei, fill=True, color="blue")

            elif (idx1,idx2) in obstacle_loc_lst:
                sq = patches.Rectangle((xi, yi), wid, hei, fill=True, color="red")
            elif (idx1,idx2) == optimal_exit:
                sq = patches.Rectangle((xi, yi), wid, hei, fill=True, color="black")
            elif (idx1,idx2)  in maze_exits_suboptimal:
                sq = patches.Rectangle((xi, yi), wid, hei, fill=True, color="yellow")
            else:
                sq = patches.Rectangle((xi, yi), wid, hei, fill=True, color="green")
            ax.add_patch(sq)

    pc = coll.PatchCollection(pat)
    ax.add_collection(pc)
    ax.relim()
    ax.autoscale_view()
    plt.axis('off')
    plt.show(block=False)
    plt.pause(pause)
    #plt.savefig('test.png', dpi=90)


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
    
    
    reward = 0
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
    
    if(r_type_norm):
        reward= 0
    else:
        reward = -1
    
    if(0 <= next_state[0] <= grid_size-1 and  0 <= next_state[1] <= grid_size-1 ):
        return list(next_state),reward,0

    else:
        next_state = state
        return list(next_state),reward,0

def QL_episode(Q, exploration, state=None, ):
    # Perform Q learning episode (pg. 131)
    # e_greedy input: 1 if using epsilon-greedy. 0 if using random uniform policy
    if (state is None):
        # state= list(nr.randint(0, high= grid_size, size= (1,4))[0])
        state = [0, 0]
    STATE_HEAT_MAT[state[0], state[1]] += 1


    G = 0
    for x in range(30000):
        if not using_cust:
            if exploration == "epsilonGreedy":
                a = epsilon_greedy(state, Q, p2)
            elif exploration == "UCB":
                a = UCB(state,Q,p2)
            elif exploration == "pursuit":
                a = Pursuit(state, Q,p2)
            elif exploration == "softmax":
                a = softmax(state, Q,p2)
        else:

            if exploration == "epsilonGreedy":
                a = epsilon_greedy(state, Q, cparams.e)
            elif exploration == "UCB":
                a = UCB(state,Q,cparams.e)
            elif exploration == "pursuit":
                a = Pursuit(state, Q,cparams.e)
            elif exploration == "softmax":
                a = softmax(state, Q,cparams.e)

        state_act = tuple(state + [a])
        next_state, r, terminate = step_model(tuple(state), a)
        STATE_HEAT_MAT[next_state[0], next_state[1]] += 1


        G += r
        if terminate:
            Q[state_act] = Q[state_act] + alpha * (r - Q[state_act])
            break
        else:
            Q[state_act] = Q[state_act] + alpha * (r + gamma * np.max(Q[tuple(next_state)]) - Q[state_act])
            state = next_state

    return Q, G, x


def epsilon_greedy(state, Q, alpha):
    # Given a state space, action space, and Q matrix, perform epsilon greedy to choose next action.
    # Ties between the best actions are broken randomly.

    best_action = nr.choice(np.flatnonzero(Q[tuple(state)] == Q[tuple(state)].max()))
    if (nr.rand() < 1 - alpha):
        a = best_action
    else:
        a = nr.choice(np.delete(A, best_action))

    return a

def UCB(state,Q,C):
    #First Select all actions once.
    actions_count = UCB_param.QA_COUNT_UCB[state[0],state[1]] #1x4 row
    if 0 in actions_count:
        a= np.where(actions_count == 0)[0][0] #pick one of the action that were not executed yet.
        UCB_param.QA_COUNT_UCB[state[0], state[1], a] += 1  # Update the action count array.
        return a

    else:
        action_values_arr = []#1,4 array
        for i in range(4):
            ratio = np.sqrt(2*np.log(np.sum(actions_count))/actions_count[i])
            bonus = 100*C*ratio
            total = Q[state[0],state[1],i] + bonus
            action_values_arr.append(total)


    a = np.argmax(action_values_arr)
    UCB_param.QA_COUNT_UCB[state[0], state[1],a] +=1 #Update the action count array.
    return a

def Pursuit(state,Q,B):
    actions_values = Q[state[0],state[1]] #1x4 row
    selection_probs = Pursuit_param.SelectionProb[state[0],state[1]]
    new_select_prob = []
    for i in range(4):
        new_select_prob.append(selection_probs[i]+B*(0-selection_probs[i]))

    max_act = np.argmax(actions_values)
    new_select_prob[max_act] = selection_probs[max_act]+B*(1-selection_probs[max_act])
    
    Pursuit_param.SelectionProb[state[0],state[1]] = new_select_prob
    
    #print(new_select_prob)
    #print("Probs: ", np.sum(new_select_prob))
    a= np.random.choice(A, 1, p=new_select_prob)[0]

    return a


def softmax(state,Q, temp):
    act_vals= Q[tuple(state)]
    num= np.exp(act_vals / temp)
    act_dist= num / np.sum(num)
    return nr.choice(A, p= act_dist)


def cust1(G, SD):
  
    if(G > cparams.m):
        cparams.m= G
        cparams.e= cparams.hb
        cparams.sdm=0
    
    else:
        if(SD < cparams.sdm):
            cparams.e= cparams.e - cparams.step
        else:
            cparams.sdm= SD
            cparams.e= cparams.e + cparams.step
    
    cparams.last_sd= SD
    if cparams.e > cparams.hb:
        cparams.e = cparams.hb
    elif(cparams.e < cparams.lb):
        cparams.e = cparams.lb
    
    

def teach_model(Q, N_episodes, exploration= "epsilonGreedy"):
    # Teach the model on N_episodes.
    # MC = 1 Use Monte-Carlo, MC = 0 Use Q-Learning
    # e_greedy input: 1 if using epsilon-greedy. 0 if using random uniform policy
    G_arr = []
    steps_arr= []
    sd_arr=[]
    print("Using: ", exploration)
    
    for n in range(N_episodes):
        if n%100 == 0:
            print("Episode: ", n)
        Q, G, steps = QL_episode(Q, exploration)
        if using_cust:
            cust1(G, np.std(Q.flatten()))
        sd_arr.append(np.std(Q.flatten()))
        if n>=2900:
            G_arr.append(G)
            steps_arr.append(steps)
            
    return G_arr, steps_arr, sd_arr

def test_model(Q, greedy= True):
    state = start_state
    drawWorld(map_size=grid_size, agent_loc=state, obstacle_loc_lst=obstacle_list, optimal_exit=terminal_list[-1],
              maze_exits_suboptimal=terminal_list[0:-1], pause=pause)
    plt.savefig("test_images/fig0")
    i=1

    while state != (-1,-1):
        if greedy is True:
            a = nr.choice(np.flatnonzero(Q[tuple(state)] == Q[tuple(state)].max()))
        else:
            x=2 #put something different here.

        row = state[0]
        col = state[1]
        
        if (nr.rand() >= 0.70):
            a = nr.choice(np.delete(A, a))

        if a == 0:
            next_state = (row + 1, col)
        if a == 1:
            next_state = (row - 1, col)
        if a == 2:
            next_state = (row, col + 1)
        if a == 3:
            next_state = (row, col - 1)

        if (next_state in terminal_list):
            break

        if (next_state in obstacle_list):
            next_state = state

        state = next_state
        drawWorld(map_size=grid_size, agent_loc=state, obstacle_loc_lst=obstacle_list,
                  optimal_exit=terminal_list[-1], maze_exits_suboptimal=terminal_list[0:-1], pause=pause)
        plt.savefig("test_images/fig"+str(i))
        i +=1


def search_params(search_options):
    # Get the exploration parameters from Table III
    # exploration_params[type][column # from table III]
    exploration_params= {'UCB' : {0: (.16,0.007), 1: (.1, .007), 2: (.16,.69), 3: (.1, 1.19)},
             'epsilonGreedy' : {0: (.95,.29), 1: (.83, .36), 2: (.59,.20), 3: (.65,.15)},
             'softmax' : {0: (.16,.12), 1: (.1,.12), 2: (.1,8.79), 3: (.1,8.79)},
             'pursuit' : {0: (.95,.007), 1: (.95,.007), 2: (.65, .007), 3: (.59, .007)}}
    
    if(search_options[1]):
        if(search_options[2]):
            p_type= 0
        else:
            p_type= 1
    else:
        if(search_options[2]):
            p_type= 2
        else:
            p_type= 3
    
    return exploration_params[search_options[0]][p_type]

def init_Q(search_options):
    if(not search_options[2]):
        return np.zeros([grid_size, grid_size, 4])
    else:
        if(search_options[1]):
            return np.ones([grid_size, grid_size, 4])
        else:
            return 200* np.ones([grid_size, grid_size, 4])
    
#UCB Parameters
class UCB_Params():
    def __init__(self):
        self.QA_COUNT_UCB = np.zeros([grid_size, grid_size, 4]) #This is a variable used during UCB

#pursuit Parameters
class pursuit_Params():
    def __init__(self):
        self.SelectionProb = np.ones([grid_size, grid_size, 4]) * .25 #This is a variable used during UCB
        
class cust1_Params():
    def __init__(self, A, ltype, norm, ):
        
        exploration_limits= {'UCB' : {0: (.2,2.5), 1: (.002, .025)},
             'epsilonGreedy' : {0: (0,1), 1: (0,1)},
             'softmax' : {0: (5,15), 1: (.01,1.5)},
             'pursuit' : {0: (1e-5,.1), 1: (1e-5,.1)}}
        
        bounds= exploration_limits[ltype][norm]
        self.lb= bounds[0]
        self.hb= bounds[1]
        
        self.step= (self.hb- self.lb) * A
        
        self.m= 0
        self.n= 0
        self.e= bounds[1]
        self.sdm= 0
        
        
        



# Parameters defined globally
grid_size = 10
start_state = (0, 0)

terminal_list = [(9,0), (0,9), (7, 7)]
rewards_list_norm = [0.5, 0.5, 1]
rewards_list= [50,50,200]
A = np.array([0, 1, 2, 3])
obstacle_list = [(7, 6), (5, 8), (9, 3), (8, 4), (5, 6), (5, 7), (2, 9), (7, 6), (6, 6)]


gamma = 0.98
pause = 0.01  # seconds

RETURNS_ARR = []
exploration = ["epsilonGreedy","UCB","softmax"]
exploration= ["pursuit"]
STATE_HEAT_MAT = np.zeros((grid_size,grid_size)) #Used for heat map

random_maze = pickle.load(open("valid_courses.p", "rb"))
rows=[]
total_std=[]
for e in exploration:
    for n in [True, False]:
        for o in [True, False]:
            for using_cust in [False]:
                results=[]
                #**** THIS SHOULD BE THE ONLY VARIABLE YOU HAVE TO MODIFY FOR ALL OPTIONS IN 
                #**** TABLE III. [0]: Exploration type. [1]: Normalized (bool). [2]: Optimistic (bool)
                SEARCH_OPTIONS= [e, n, o]
                
                alpha, p2= search_params(SEARCH_OPTIONS)
                r_type_norm= SEARCH_OPTIONS[1]
                
                UCB_param = UCB_Params()
                Pursuit_param = pursuit_Params()
                
                cparams= cust1_Params(0.05, e, n)
                
                total_R =[]
                total_steps= []
                for i in range(1):
                    print("Maze Number: ", i)
                    drawWorld(map_size=grid_size, agent_loc=start_state, obstacle_loc_lst=obstacle_list,optimal_exit=terminal_list[-1],maze_exits_suboptimal=terminal_list[0:-1], pause = pause)
                    plt.close()
                    for k in range(1):
                        print("Trial Number: ", k+1, "/3")
                        Q_init= init_Q(SEARCH_OPTIONS)
                
                        R_arr, steps_arr, sd_arr = teach_model(Q_init, 3000,exploration=SEARCH_OPTIONS[0])
                        
                        total_R= total_R + R_arr
                        total_steps = total_steps + steps_arr
                
                total_std.append(sd_arr)
                RETURNS_ARR= np.array(total_R).flatten()
                STEPS_ARR= np.array(total_steps).flatten()

                print("Size: ", len(RETURNS_ARR))
                print("Average: ", np.average(RETURNS_ARR))
                print("STD: ", np.std(RETURNS_ARR))
                print("Avg. Steps: ", np.average(STEPS_ARR))

                results.append(e)
                results.append(n)
                results.append(o)

                results.append(np.average(RETURNS_ARR))
                results.append(np.std(RETURNS_ARR))
                results.append(np.average(STEPS_ARR))

                rows.append(results)

                # Create Figure level:
                heat_map_name = e
                if using_cust is True:
                    heat_map_name += "_Cust"
                else:
                    heat_map_name += "_NoCust"

                if n is True:
                    heat_map_name += "_norm"
                else:
                    heat_map_name += "_NoNorm"
                if o is True:
                    heat_map_name += "_Opt"
                else:
                    heat_map_name += "_NoOpt"
                heat_map_name += ".png"
                STATE_HEAT_MAT = np.rot90(STATE_HEAT_MAT)
                ax = sns.heatmap(STATE_HEAT_MAT, linewidth=0.5)
                plt.savefig("walk_heatMaps/maze1/" + heat_map_name, dpi=90)
                plt.close()
                STATE_HEAT_MAT = np.zeros((grid_size, grid_size))  # Used for heat map
#
# results_df= pd.DataFrame(rows, columns= ['Exploration Type','Normalized', 'Optimistic', 'Annealing Param', 'Mean Reward', 'SD', 'Mean Steps', 'SD Steps'])
#
# print(results_df)