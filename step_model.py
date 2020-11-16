def step_model(state, a):
    #Take a state and an action, return the next state, reward, and whether
    #the terminal state was entered
    row= state[0]
    col= state[1]
    
    if a == 0:
        next_state= (row + 1, col)
    if a == 1:
        next_state= (row - 1, col)
    if a == 2:
        next_state= (row, col + 1)
    if a == 3:
        next_state= (row, col - 1)
    
    if(next_state in terminal_list):
        reward= rewards_list[terminal_list.index(next_state)]
        next_state = (-1,-1)
        return next_state, reward, 1
    
    if(next_state in obstacle_list):
        next_state= state
        return next_state,0,0
    
    if(0 <= next_state[0] <= grid_size-1 and  0 <= next_state[1] <= grid_size-1 ):
        return next_state,0,0

    else:
        next_state = state
        return next_state,0,0


#Parameters defined globally 
grid_size= 10
start_state= (0,0)

terminal_list=[(0,9),(9,0),(7,7)]
rewards_list=[50,50,200]
obstacle_list=[(3,0),(4,0),(8,0),(9,2),(6,4),(7,5),(3,9),(4,8),(9,8)]

A= [0,1,2,3]

print(step_model((0,0),0))