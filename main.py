'''
Date: 11/17/20

— Add new exploration (UCB-1, Pursuit)
— Add map.
— Add testing.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
import numpy.random as nr
import argparse
import time
import sys


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
    # Take a state and an action, return the next state, reward, and whether
    # the terminal state was entered

    if (nr.rand() >= 0.70):
        a = nr.choice(np.delete(A, a))

    row = state[0]
    col = state[1]

    if a == 0:
        next_state = (row + 1, col)
    if a == 1:
        next_state = (row - 1, col)
    if a == 2:
        next_state = (row, col + 1)
    if a == 3:
        next_state = (row, col - 1)

    if (next_state in terminal_list):
        reward = rewards_list[terminal_list.index(next_state)]
        next_state = (-1, -1)
        return list(next_state), reward, 1

    if (next_state in obstacle_list):
        next_state = list(state)
        return list(next_state), 0, 0

    if (0 <= next_state[0] <= grid_size - 1 and 0 <= next_state[1] <= grid_size - 1):
        return list(next_state), 0, 0

    else:
        next_state = list(state)
        return list(next_state), 0, 0


def QL_episode(Q, exploration, state=None, ):
    # Perform Q learning episode (pg. 131)
    # e_greedy input: 1 if using epsilon-greedy. 0 if using random uniform policy
    if (state is None):
        # state= list(nr.randint(0, high= grid_size, size= (1,4))[0])
        state = [0, 0]

    G = 0
    for x in range(1000):
        if exploration == "epsilongGreedy":
            a = epsilon_greedy(state, Q)
        elif exploration == "UCB":
            a = UCB(state,Q)

        state_act = tuple(state + [a])
        next_state, r, terminate = step_model(tuple(state), a)

        G += r
        if terminate:
            Q[state_act] = Q[state_act] + alpha * (r - Q[state_act])
            break
        else:
            Q[state_act] = Q[state_act] + alpha * (r + gamma * np.max(Q[tuple(next_state)]) - Q[state_act])
            state = next_state

    return Q, G


def epsilon_greedy(state, Q):
    # Given a state space, action space, and Q matrix, perform epsilon greedy to choose next action.
    # Ties between the best actions are broken randomly.

    best_action = nr.choice(np.flatnonzero(Q[tuple(state)] == Q[tuple(state)].max()))
    if (nr.rand() < 1 - epsilon + (epsilon / grid_size)):
        a = best_action
    else:
        a = nr.choice(np.delete(A, best_action))

    return a

def UCB(state,Q):
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
            bonus = 100*UCB_param.C*ratio
            total = Q[state[0],state[1],i] + bonus
            action_values_arr.append(total)


    a = np.argmax(action_values_arr)
    UCB_param.QA_COUNT_UCB[state[0], state[1],a] +=1 #Update the action count array.
    return a



def teach_model(Q, N_episodes, exploration= "epsilongGreedy"):
    # Teach the model on N_episodes.
    # MC = 1 Use Monte-Carlo, MC = 0 Use Q-Learning
    # e_greedy input: 1 if using epsilon-greedy. 0 if using random uniform policy
    G_arr = []
    print("Using: ", exploration)

    try:
        for n in range(N_episodes):
            if n%100 == 0:
                print("Episode: ", n)
            Q, G = QL_episode(Q, exploration)
            G_arr.append(G)
    finally:
        G = 0
        G_avg_arr = []
        for n, g in enumerate(G_arr):
            G += g
            if (n > 1 and n % 100 == 0):
                G_avg_arr.append(G / 100)
                G = 0
        plt.plot(range(0, len(G_avg_arr)), G_avg_arr)
        plt.show()

    return Q





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


#UCB Parameters
class UCB_Params():
    def __init__(self):
        self.QA_COUNT_UCB = np.zeros([grid_size, grid_size, 4]) #This is a variable used during UCB
        self.C = 1

# Parameters defined globally
grid_size = 10
start_state = (0, 0)

terminal_list = [(9,0), (0,9), (7, 7)]
rewards_list = [0.5, 0.5, 1]
obstacle_list = [(0,3),(0,4),(9,3),(8,4),(4,6),(5,7),(2,9),(0,8),(8,9)]
A = np.array([0, 1, 2, 3])


#UCB Parameters:
UCB_param = UCB_Params()

epsilon = 0.1
alpha = 0.1
gamma = 0.98
pause = 0.01  # seconds

if __name__ == "__main__":

    #drawWorld(map_size=grid_size, agent_loc=start_state, obstacle_loc_lst=obstacle_list,optimal_exit=terminal_list[-1],maze_exits_suboptimal=terminal_list[0:-1], pause = pause)

    #exploration options
    exploration = ["epsilongGreedy","UCB"]
    Q = teach_model(np.zeros([grid_size, grid_size, 4]), 50000,exploration=exploration[1])
    test_model(Q)






