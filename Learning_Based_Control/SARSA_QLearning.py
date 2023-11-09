from random import randint
from copy import deepcopy
import numpy as np
import random
import matplotlib.pyplot as plt

##SOURCES
# https://www.geeksforgeeks.org/sarsa-reinforcement-learning/#
# https://www.geeksforgeeks.org/q-learning-in-python/

#%%
class gridWorld:
    def __init__(self):
        self.init_door=[9,1]
        self.agent=None
        self.door=None

    def reset(self):
        self.door=self.init_door
        position=[-1,-1]
        while not self.isValid(position):
            position[0]=randint(0,9)
            position[1]=randint(0,4)
        self.agent=position
        return deepcopy(self.agent)
    
    def reset_pos(self,position):
        self.agent=position
        return deepcopy(self.agent)

    def take_action(self,action,position):
        x,y=position
        if action=="up":
            y+=1
        if action=="down":
            y-=1
        if action=="right":
            x+=1
        if action=="left":
            x-=1
        if self.isValid([x,y]):
            return [x,y]
        return position

    def step(self,action,rng_door=False):
        self.agent=self.take_action(action,self.agent)
        if rng_door:
            rng_action=["up","down","left","right"][randint(0,3)]
            self.door=self.take_action(rng_action,self.door)
        if self.agent[0]==self.door[0] and self.agent[1]==self.door[1]:
            reward=20
        else:
            reward=-1
        return deepcopy(self.agent),reward

    def isValid(self,position):
        x,y=position
        if x<0 or x>9:      #out of x bounds
            return False
        if y<0 or y>4:      #out of y bounds
            return False
        if x==7 and y<3:    #if door
            return False
        return True
    
def pos_to_matrix(position):
    x,y=position
    matrix_index = x*5 + (4-y) # 4-y reverses direction of column counts
    return matrix_index

def matrix_to_pos(matrix_index):
    x = matrix_index // 5  # Integer division to get the x coordinate
    y = matrix_index % 5    # Modulo operation to get the y coordinate
    return x, 4-y 

def get_action(state):
    index = pos_to_matrix(state)
    global epsilon
    global q_table
    if np.random.uniform(0,1) > epsilon: # does it exceed probability to accept best solution?
        # Take random action
        action = random.randint(0, 4)
    else:
        # Do what the policy states
        action = np.argmax(q_table[index, :]) #returns indice of max value, or best action
    return action

def get_movement_array(q_table,movement_dim):
    best_moves_list = []
    for row in q_table:
        best_move = np.argmax(row)
        action = ["up","down","left","right","stay"][best_move]
        best_moves_list.append(action)
    movement_map = np.empty(movement_dim, dtype='U5')
    for index, move in enumerate(best_moves_list):
        col, row = matrix_to_pos(index) # get matrix position
        movement_map[4-row,col] = move
    for i in range(2,5):
        movement_map[i,7] = 'WALL'
    return movement_map

def plot_scatter(x,y,title):
    # Create a scatter plot
    plt.scatter(x, y, color='black')
    plt.plot(x, y, color='blue', linestyle='-', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Moving Average Over 30 Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def increasing_epsilon(intial_epsilon,iteration,total_iterations,
                      max_epsilon=.99,percent_til_flat=0.70):
    p95 = int(total_iterations*percent_til_flat) # to set linear function
    if (iteration  /p95) >=1:
        return max_epsilon # never get to 1
    else:
        adj_range = (max_epsilon - intial_epsilon) * (iteration / p95)
        return intial_epsilon + adj_range
    
def alpha_decay(intial_alpha,iteration,total_iterations,
                      min_alpha=.01,percent_til_flat=0.90):
    p95 = int(total_iterations*percent_til_flat) # to set linear function
    if (iteration  /p95) >=1:
        return min_alpha # never get to 1
    else:
        adj_range = (intial_alpha-min_alpha) * (iteration / p95)
        return intial_alpha - adj_range

def test_q_table_performance(q_table):
    # Loops through every state possible and sums up rewards from ideal q_table
    total_reward = []
    env=gridWorld() # intialize gridworld class
    state = env.reset() #intialize door position
    for i in range(50): # loop through every state possible
        if i in (37, 38, 39): # these values are the 'wall'
            continue  # Skip these numbers
        state = env.reset_pos(matrix_to_pos(i)) # loop through every possible start position
        movement_index = np.argmax(q_table[i, :]) # get best first action from start position
        action = ["up","down","left","right","stay"][movement_index] # take first step
        end_reward = 0 # start total reward with 0
        for step in range(20): # do 20 steps
            #the action is taken, a reward and new state is returned
            next_state, reward=env.step(action)  
            # note: use env.step(action,rng_door=True) for part 2
            # based on new state, choose best action
            next_movement_index = np.argmax(q_table[pos_to_matrix(next_state), :]) 
            #learner chooses one of these actions
            next_action=["up","down","left","right","stay"][next_movement_index] 
            action = next_action # store action to be used in the loop
            end_reward += reward # add up reward
        total_reward.append(end_reward) # record reward for that state.
    return sum(total_reward) # return sum of rewards to represent how well that policy did

def moving_average(list_to,window=30):
    final_moving_average_list = []
    iteration_count = len(list_to) - window
    for i in range(iteration_count):
        average_window = 0
        for num in range(0+i,window+i):
            average_window += list_to[num]
        final_moving_average_list.append(average_window / window)
    return final_moving_average_list
           
    
#%% Set Parameters
intial_epsilon = 0.1 # epislon greedy probability to accept best solution 
# ^ start low for exploration, increase
episodes = 500000 # total number of iterations
intial_alpha = 0.9 # learning rate. Start high, and decay.
gamma = 0.9 # discount factor, higher prioritizes future rewards
time_steps = 20 # steps allowed within one attempt, set at 20

#%% SARSA LEARNING
def q_table_sarsa_update(state_pos, current_action, reward,
                         future_state_pos, future_action):
    # Declare global variables used
    global q_table
    global gamma
    global alpha
    current_state = pos_to_matrix(state_pos)
    future_state = pos_to_matrix(future_state_pos)
    # Q(s, a) = Q(s, a) + alpha * (r + gamma * Q(s’, a’) – Q(s, a))
    predicted_state = q_table[current_state,current_action]
    update = reward + gamma * q_table[future_state,future_action] - predicted_state
    q_table[current_state, current_action] = (q_table[current_state, current_action] + 
                                              alpha * update)

# Intialize Q matrix with observation space x action space
q_table = np.random.rand(5*10,5) * 0.01 

env=gridWorld() # intialize gridworld class
q_table_performance_list = [] # intialize moving average list for plotting
for learning_epoch in range(episodes):
    # slowly increase probability to accept best solution
    epsilon = increasing_epsilon(intial_epsilon,learning_epoch,episodes) 
    alpha = alpha_decay(intial_alpha,learning_epoch,episodes) # slowly decay
    state=env.reset() #every episode, reset the environment to the original configuration
    movement_index = get_action(state) # 0,1,2,3 representing up, down, left, right
    action = ["up","down","left","right","stay"][movement_index] # take first step
    
    for time_step in range(time_steps): # take 20 steps
    #the action is taken, a reward and new state is returned
        next_state, reward=env.step(action)  
        #note: use env.step(action,rng_door=True) for part 2
        # get movement index to decide action, epsilon greedy
        next_movement_index = get_action(next_state) 
        #learner chooses one of these actions
        next_action=["up","down","left","right","stay"][next_movement_index] 
        # Update q_table 
        q_table_sarsa_update(state,movement_index,reward,next_state,next_movement_index)
        # Move next values to current values for next iteration
        movement_index = next_movement_index
        action = next_action
        state = next_state
    if learning_epoch % 1000 == 0:
        q_table_performance = test_q_table_performance(q_table)
        q_table_performance_list.append(q_table_performance)
        
sarsa_learning_movement = get_movement_array(q_table,(5,10))
moving_average_q_table_performance_list = moving_average(q_table_performance_list)

iteration_num = [i*1000 + 30000 for i in range(len(moving_average_q_table_performance_list))]

plot_scatter(iteration_num,moving_average_q_table_performance_list,
              'SARSA learning, Q table performance through all possible starting positions')

#%% Q LEARNING
def q_table_qlearning_update(state_pos, current_action, reward,
                          future_state_pos, future_action):
    # Declare global variables used
    global q_table
    global gamma
    global alpha
    current_state = pos_to_matrix(state_pos)
    future_state = pos_to_matrix(future_state_pos)
    # Q(s, a) = Q(s, a) + alpha * (r + gamma * maxaQ(s’, a) – Q(s, a))
    predicted_state = q_table[current_state,current_action]
    max_future_q_value = np.max(q_table[future_state, :]) # Find the max Q-value for next state
    update = reward + gamma * max_future_q_value - predicted_state
    q_table[current_state, current_action] = (q_table[current_state, current_action] + 
                                              alpha * update)
    
# Intialize Q matrix with observation space x action space
q_table = np.random.rand(5*10,5) * 0.01 

env=gridWorld() # intialize gridworld class
q_table_performance_list = [] # intialize moving average list for plotting
for learning_epoch in range(episodes):
    state=env.reset() #every episode, reset the environment to the original configuration
    movement_index = get_action(state) # 0,1,2,3 representing up, down, left, right
    action = ["up","down","left","right","stay"][movement_index] # take first step
    
    for time_step in range(time_steps): # take 20 steps
        #the action is taken, a reward and new state is returned
        next_state, reward=env.step(action,rng_door=False)  
        # note: use env.step(action,rng_door=True) for part 2
        # get movement index to decide action, epsilon greedy
        next_movement_index = get_action(next_state) 
        #learner chooses one of these actions
        next_action=["up","down","left","right","stay"][next_movement_index] 
        # Update q_table 
        q_table_qlearning_update(state,movement_index,reward,next_state,next_movement_index)
        # Move next values to current values for next iteration
        movement_index = next_movement_index
        action = next_action
        state = next_state
    if learning_epoch % 1000 == 0:
        q_table_performance = test_q_table_performance(q_table)
        q_table_performance_list.append(q_table_performance)
        
q_learning_movement = get_movement_array(q_table,(5,10))

moving_average_q_table_performance_list = moving_average(q_table_performance_list)
iteration_num = [i*1000 + 30000 for i in range(len(moving_average_q_table_performance_list))]
plot_scatter(iteration_num,moving_average_q_table_performance_list,
              'Q-learning, Q table performance through all possible starting positions')
