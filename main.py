
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
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




class Parameters():
    def __init__(self):
        self.mapSize = 10


if __name__ == "__main__":
    map_s = 10
    agent_loc= (0,0)
    obstacle_loc_lst = [(0,3),(0,4),(9,3),(8,4),(4,6),(5,7),(2,9),(0,8),(8,9)]
    optimal_exit = (7,7)
    maze_exits_suboptimal = [(9,0), (0,9)]
    pause = 10 #seconds

    drawWorld(map_size=map_s, agent_loc=agent_loc, obstacle_loc_lst=obstacle_loc_lst,optimal_exit=optimal_exit,maze_exits_suboptimal=maze_exits_suboptimal, pause = pause)







