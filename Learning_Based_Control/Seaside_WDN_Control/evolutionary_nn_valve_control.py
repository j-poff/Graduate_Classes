import numpy as np
import os
import matplotlib.pyplot as plt  
import wntr
import random
import warnings
import json
import heapq
#%% Functions

class OneLayerNN:
    # Expects training to be a 1 or 0, 1 if pa
    def __init__(self, input_dim, hidden_dim, output_dim, weight_initial=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_initial = weight_initial
        
        # Initialize weights and biases for input and hidden layer
        self.w_input_hidden = np.random.uniform(-self.weight_initial, self.weight_initial, size=(input_dim, hidden_dim))
        self.b_input_hidden = np.zeros((1, hidden_dim))
        # Initialize weights and biases for hidden and output layers
        self.w_hidden_output = np.random.uniform(-self.weight_initial, self.weight_initial, size=(hidden_dim, output_dim))
        self.b_hidden_output = np.zeros((1, output_dim))     
        
    def forward(self, x):
        # Forward propagation through the network
        transformed_input_hidden = np.dot(x, self.w_input_hidden) + self.b_input_hidden
        hidden_activations = self._relu(transformed_input_hidden)
        self.hidden_activations = hidden_activations
    
        # Calculate the output of the two output layers simultaneously
        transformed_hidden_outputs = (np.dot(hidden_activations, self.w_hidden_output) 
                                      + self.b_hidden_output)
        output_activations = self._sigmoid(transformed_hidden_outputs)

        return output_activations
    
    def mutate_and_create_copy(self, mutation_rate, mutate_distribution_int=0.1):
        # Create a new instance with the same initial parameters
        mutated_copy = OneLayerNN(self.input_dim, self.hidden_dim, self.output_dim)
        # Copy the weights and biases from the original instance
        mutated_copy.w_input_hidden = np.copy(self.w_input_hidden)
        mutated_copy.b_input_hidden = np.copy(self.b_input_hidden)
        mutated_copy.w_hidden_output = np.copy(self.w_hidden_output)
        mutated_copy.b_hidden_output = np.copy(self.b_hidden_output)
        # Apply mutation to the copy
        mutated_copy._mutate(mutation_rate, mutate_distribution_int)
        return mutated_copy
    
    def _mutate(self, mutation_rate, mutate_distribution_int):
        # Go through all layers and mutate! Some according to mutation rate
        self._mutate_parameters(self.w_input_hidden, mutation_rate, mutate_distribution_int)
        self._mutate_parameters(self.b_input_hidden, mutation_rate, mutate_distribution_int)
        self._mutate_parameters(self.w_hidden_output, mutation_rate, mutate_distribution_int)
        self._mutate_parameters(self.b_hidden_output, mutation_rate, mutate_distribution_int)
        return 

    def _mutate_parameters(self,parameters,mutation_rate, mutate_distribution_int):
        # Create mask
        mutation_mask = np.random.rand(*parameters.shape) < mutation_rate
        # Mutate by random noise from normal distribution
        parameters += mutation_mask * np.random.normal(0, mutate_distribution_int, size=parameters.shape)

    def predict(self, x):
       predictions = self.forward(x)
       return predictions

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _relu(self, x):
        return np.maximum(0, x)

def get_fitness(wn, modified_nn, initial_pressures, episodes=10, scaling=1000):
    wn.options.hydraulic.demand_model = 'PDD' # pressure driven analysis (better for leaky pipes)
    wn.options.time.duration = 0 # steady state, snapshot
    wn.options.hydraulic.headloss = 'D-W' # Use hazen williams
    valves = wn.valve_name_list
    setting_list = []
    fitness_over_episodes = []
    
    predicted_valve_control = modified_nn.forward(initial_pressures)
    valve_settings = predicted_valve_control.tolist()[0]
    
    for episode in range(episodes):
        episode_setting = []
        # Set all valves to specified setting.
        for index, valve in enumerate(valves):
            valve_obj = wn.get_link(valve) # links are valves
            valve_obj.initial_setting = valve_settings[index] * scaling
            episode_setting.append(valve_settings[index] * scaling)
            
        sim_in = wntr.sim.EpanetSimulator(wn)
        # sim_in = wntr.sim.WNTRSimulator(wn)
        results_in = sim_in.run_sim() # by default, this runs EPANET 2.2.0
        pressure_in = results_in.node['pressure']
        initial_pressures_in = pressure_in.iloc[0] #pressures at 0
        
        episode_fitness = 0
        for pressure in initial_pressures_in:
            if 30 <= pressure <= 100:
                episode_fitness += 1
                
        fitness_over_episodes.append(episode_fitness)   
        predicted_valve_control = modified_nn.forward(initial_pressures_in)
        valve_settings = predicted_valve_control.tolist()[0]
        setting_list.append(episode_setting) #store all the settings
    
    reward = [item*((index+1)/episodes) for index, item, in enumerate(fitness_over_episodes)]
    return sum(reward), setting_list

def moving_average(list_to,window=30):
    final_moving_average_list = []
    iteration_count = len(list_to) - window
    for i in range(iteration_count):
        average_window = 0
        for num in range(0+i,window+i):
            average_window += list_to[num]
        final_moving_average_list.append(average_window / window)
    return final_moving_average_list

def plot_scatter(x,y,title,y_label='Value'):
    # Create a scatter plot
    plt.figure()
    plt.scatter(x, y, color='black')
    plt.plot(x, y, color='blue', linestyle='-', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

def add_valve_between_pipe(wn, pipe1_name, pipe2_name, valve_type, setting):
    # Find shared junction
    if (wn.get_link(pipe1_name).end_node_name == wn.get_link(pipe2_name).end_node_name) or (
            wn.get_link(pipe1_name).end_node_name == wn.get_link(pipe2_name).start_node_name):
        shared_junction = wn.get_node(wn.get_link(pipe1_name).end_node_name)
        shared_1 = '1_end'
    elif (wn.get_link(pipe1_name).start_node_name == wn.get_link(pipe2_name).end_node_name) or (
            wn.get_link(pipe1_name).start_node_name == wn.get_link(pipe2_name).start_node_name):
        shared_junction = wn.get_node(wn.get_link(pipe1_name).start_node_name)
        shared_1 = '1_start'
    else:
        raise ValueError(f'Input pipes {pipe1_name} and {pipe2_name} do not share a matching junction')
    # Create new junction
    shared_junction_elevation = shared_junction.elevation
    new_junction_name = f'V-{pipe1_name}_{pipe2_name}'
    wn.add_junction(new_junction_name, base_demand=0, elevation=shared_junction_elevation)
    new_junction_object = wn.get_node(new_junction_name)
    # Disconnect first pipe from shared and connect it to the new link
    if shared_1 == '1_end':
        wn.get_link(pipe1_name).end_node = new_junction_object
    elif shared_1 == '1_start':
        wn.get_link(pipe1_name).start_node = new_junction_object
    # Create new valve
    valve_name = f'{valve_type}-{pipe1_name}_{pipe2_name}'
    diameter = wn.get_link(pipe2_name).diameter  # Set diameter to the same as the disconnected pipe
    wn.add_valve(valve_name, new_junction_object.name, shared_junction.name, diameter=diameter, 
                  valve_type=valve_type, minor_loss=0.0, initial_setting=setting)
    return

pipe_pair_for_valve = [ [772,785],
                       [491,493],
                       [684,810],
                       [39,38],
                       [452,451],
                       [297,425],
                       [428,426],
                       [139,86],
                       [1,5],
                       [55,53],
                       [819,651],
                       [330,346],
                       [440,441],
                       [508,526],
                       [510,633],
                       [786,830]
    ]

warnings.filterwarnings("ignore") # ignore all warnings for low pressure simulations   
#%% Set up loop!
disaster_rt_vals = [250]

for rt_val in disaster_rt_vals:
    
#%% set variables
    # rt_val = 1000 # 100,250,500,1000,2500,5000,10000 #FIXME
    scaling = 500 # Max flow should be what? GPM
    headloss = 'D-W'
    
    # Search variables
    hidden_layers = 32 # how many hidden layers in neural network
    n_population = 30 # how many are in the population
    keep_best = 25 # number of best performing networks to keep, will generate random others to fill
    mutate_number = 2 # number of networks that should be mutated for each iterations
    mutation_rate = .20 # mutate random 20% of network parameters?
    modify_by_normal_dist = 0.1 # mutate within normal distribution of?
    weight_initialization = 0.1 # initialize weights and biases between -x and x
    epsilon = .99 # Start high to explore, and decay
    epsilon_decay = .995 # how much you decay by
    min_epsilon = 0.03 # the lowest epsilon you want
    iterations = 10000
    episode_length = 1 # 1 seems to work best...
    moving_window = 10 # how many values to average within for plotting
    
    os.makedirs(f'results/{rt_val}yr', exist_ok=True)
    os.makedirs(f'results/{rt_val}yr/{iterations}iter', exist_ok=True)
    save_folder_path = f'results/{rt_val}yr/{iterations}iter/'
    #%% Loading Data
    wn_pre_disaster = wntr.network.WaterNetworkModel(os.getcwd() + '/src/Seaside_dummy_model.inp') #load water network
    print(wn_pre_disaster.options.hydraulic.inpfile_pressure_units)
    print(wn_pre_disaster.options.hydraulic.inpfile_units)
    wn_pre_disaster.options.hydraulic.demand_model = 'PDD' # pressure driven analysis (better for leaky pipes)
    wn_pre_disaster.options.time.duration = 0 # steady state, snapshot
    wn_pre_disaster.options.hydraulic.headloss = headloss
    sim_pre = wntr.sim.EpanetSimulator(wn_pre_disaster)
    # sim_pre = wntr.sim.WNTRSimulator(wn_pre_disaster)
    results_pre = sim_pre.run_sim() # by default, this runs EPANET 2.2.0
    pressure_pre = results_pre.node['pressure']
    flow_pre = results_pre.link['flowrate']
    initial_flow_pre = abs(flow_pre.iloc[0]* 15850.323140625002)
    initial_pressures_pre = pressure_pre.iloc[0] # / .70307 #pressures at 0
    initial_pressures_pre = initial_pressures_pre.apply(lambda x: max(x, 0)) # Set everything below 0 to 0
    # Initial Plot
    ax_pre = wntr.graphics.plot_network(wn_pre_disaster, node_attribute=initial_pressures_pre,
                                        title='Pre-Disaster Pressures without Valve Control',node_range=[0, 80],
                                        node_size=80,add_colorbar=True,node_colorbar_label='Pressure, PSI')
    plt.savefig(save_folder_path + 'pre_disaster_pressure_map.png',
                dpi=600, bbox_inches='tight')
    plt.close()
    # ax_f = wntr.graphics.plot_network(wn_pre_disaster, link_attribute=initial_flow_pre,title='Pre-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow, GPM')
    fitness_count_pre = 0
    for pressure in initial_pressures_pre:
        if 30 <= pressure <= 100:
            fitness_count_pre += 1
    
    
    wn = wntr.network.WaterNetworkModel(os.getcwd() + f'/damage_data/damaged_network_{rt_val}yr.inp') #load water network
    wn_plotting = wntr.network.WaterNetworkModel(os.getcwd() + f'/damage_data/damaged_network_{rt_val}yr.inp') #load water network
    
    # Add leaks into nodes, not carried over from inp file
    with open(os.getcwd() + f'/damage_data/leaky_node_{rt_val}yr.json', 'r') as json_file:
        leaky_node_dict = json.load(json_file) # The area in leaky_node_dict is in m^2
    for node_name, node in wn.nodes():
        if leaky_node_dict[node_name] != 0:
            node.add_leak(wn, leaky_node_dict[node_name], discharge_coeff=0.75, start_time=0, end_time=None)
            
    # Add valves in for control
    for pipe_pair in pipe_pair_for_valve:
        add_valve_between_pipe(wn, str(pipe_pair[0]), str(pipe_pair[1]), 'FCV', scaling)
    wn.options.hydraulic.demand_model = 'PDD' # pressure driven analysis (better for leaky pipes)
    wn.options.time.duration = 0 # steady state, snapshot
    wn.options.hydraulic.headloss = headloss
    valves = wn.valve_name_list
    
    # Set all FCVs to some max limit indicating they are open
    for valve in valves:
        valve_obj = wn.get_link(valve) # links are valves
        valve_obj.initial_setting = scaling
    
    # obj = wntr.epanet.InpFile()
    # obj.write(r'\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\Desktop\valve_added_network.inp',wn)
    
    # Initial simulation
    sim = wntr.sim.EpanetSimulator(wn)
    # sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim() # by default, this runs EPANET 2.2.0
    pressure = results.node['pressure']
    flow = results.link['flowrate']
    initial_pressures = pressure.iloc[0] #pressures at 0
    initial_pressures = initial_pressures.apply(lambda x: max(x, 0)) # Set everything below 0 to 0
    # Remove new junctions added to be able to graph with structure
    valve_junction_list = [item.replace('FCV', 'V') for item in valves]
    initial_pressures_plot = initial_pressures.drop(valve_junction_list)
    
    # Initial Plot
    ax_p = wntr.graphics.plot_network(wn_plotting, node_attribute=initial_pressures_plot,
                                      title='Post-Disaster Pressures without Valve Control',node_range=[0, 80],
                                      node_size=80,add_colorbar=True,node_colorbar_label='Pressure, PSI')
    plt.savefig(save_folder_path + 'post_disaster_no_valve_pressure_map.png',
                dpi=600, bbox_inches='tight')
    plt.close()
    # ax_f = wntr.graphics.plot_network(wn, link_attribute=intial_flow,title='Pre-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow')
    
    fitness_count_init = 0
    for pressure in initial_pressures:
        if 30 <= pressure <= 100:
            fitness_count_init += 1
    
    #%% Evolved Neural Network
    '''
    1. At t=0 initialize N neural networks
    2. Pick a network using e-greedy alg (e=.1)
    3. Randomly modify network parameters
    4. Use network on this agent for T steps
    5. Evaluate network performance
    6. Re-insert network into pool
    7. Remove worst network from pool
    8. Go to step 2
    '''
    ### Intialize N neural networks
    input_dim = len(initial_pressures)
    output_dim = len(valves) # each output bewteen 0 and 1 represents open and closed
    
    # Intialize the neural network population
    a_nn_population_dict = {f'nn{i}':{'nn':OneLayerNN(input_dim,hidden_layers,output_dim, weight_initialization),
                                      'fitness':0,'setting':[]} for i in range(n_population)}
    
    for key, value in a_nn_population_dict.items():
        try:
            value['fitness'], value['setting'] = get_fitness(wn, value['nn'], initial_pressures,
                                           episodes=episode_length, scaling=scaling)
        except:
            print('failed to execute sim, assigning failure')
            value['fitness'], value['setting'] = 0, ['fail' for _ in range(output_dim)]
            
    best_performer = [] # list to track performance of the controls 
    for i in range(iterations):
        print(f'Iteration {i+1}')
        # Append best performer
        fitness_list = [value['fitness'] for value in a_nn_population_dict.values()]
        best_performer.append(max(fitness_list))
        
        # Degrade epsilon
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        
        # Epsilon greedy to choose what network to mutate
        if np.random.rand() < epsilon: # less than,
            nns_to_modify = random.sample(a_nn_population_dict.keys(), mutate_number)
        else:
            nns_to_modify = heapq.nlargest(mutate_number, a_nn_population_dict, 
                                        key=lambda k: a_nn_population_dict[k]['fitness'])
            
        # Mutate and add new neural networks
        for nn in nns_to_modify:
            modified_nn = a_nn_population_dict[nn]['nn'].mutate_and_create_copy(mutation_rate, modify_by_normal_dist)
            # Calculate the fitness
            try:
                nn_fitness, settings = get_fitness(wn, modified_nn, initial_pressures,
                                                   episodes=episode_length, scaling=scaling)
            except:
                print('failed to execute sim, assigning failure')
                nn_fitness, settings = 0, ['fail' for _ in range(output_dim)]
            ### Add new neural network to dictionary
            # Get a new name for dictionary
            current_names = list(a_nn_population_dict.keys())
            counter = 0
            while True:
                if f'nn{counter}' not in current_names:
                    new_name = f'nn{counter}'
                    break
                else:
                    counter += 1
            # Add new modified nn
            a_nn_population_dict[new_name] = {'nn':modified_nn, 'fitness':nn_fitness, 'setting':settings}
            
        # Keep the top performing solutions!
        nns_to_keep = heapq.nlargest(keep_best, a_nn_population_dict, 
                                    key=lambda k: a_nn_population_dict[k]['fitness'])
        # Remove the solutions not in that list
        for key in list(a_nn_population_dict.keys()):
            if key not in nns_to_keep:
                a_nn_population_dict.pop(key)
                
        # Generate new random solutions to fill in the ones deleted
        assert n_population - keep_best > 0, 'n_population must be greater than keep_best'
        for _ in range(n_population - keep_best):
            # get unique name
            current_names = list(a_nn_population_dict.keys())
            counter = 0
            while True:
                if f'nn{counter}' not in current_names:
                    new_name = f'nn{counter}'
                    break
                else:
                    counter += 1
            # Add to dictionary
            a_nn_population_dict[new_name] = {'nn':OneLayerNN(input_dim,hidden_layers,output_dim, weight_initialization),
                                              'fitness':0,'setting':[]}
            try:
                a_nn_population_dict[new_name]['fitness'], a_nn_population_dict[new_name]['setting'] = (
                    get_fitness(wn, value['nn'], initial_pressures, episodes=episode_length, scaling=scaling))
            except:
                print('failed to execute sim, assigning failure')
                a_nn_population_dict[new_name]['fitness'] = 0
                a_nn_population_dict[new_name]['setting'] = ['fail' for _ in range(output_dim)]
                    
    # Intiate solutions
    # nn_population = [OneLayerNN(input_dim,hidden_layers,
    #                             output_dim, weight_initialization) for _ in range(n_population)]
    # for nn in nn_population: 
    #     try:
    #         # Use network on agent for t steps in each episode
    #         nn_fitness, settings = get_fitness(wn, nn, initial_pressures,
    #                                            episodes=episode_length, scaling=scaling)
    #     except:
    #         print('failed to execute sim, assigning failure')
    #         nn_fitness = -int(input_dim)
    #         settings = ['fail' for _ in range(output_dim)]
    #     nn.define_fitness(nn_fitness)
    #     nn.define_settings(settings)
        
    # best_performer = []
    
    # for i in range(iterations):
    #     print(f'Iteration {i+1}')
    #     epsilon = max(epsilon * epsilon_decay, min_epsilon)
    #     # Get index of best NN
    #     nn_fitness_list = [nn.fitness for nn in nn_population]
    #     nn_settings_list = [nn.settings for nn in nn_population]
    #     max_index = nn_fitness_list.index(max(nn_fitness_list))
    #     best_performer.append(nn_fitness_list[max_index])
          
    #     # Epsilon greedy
    #     if np.random.rand() < epsilon: # less than, 
    #         nn_to_modify = nn_population[random.randint(0, n_population-1)] # Mutate random network 
    #     else: 
    #         nn_to_modify = nn_population[max_index] # Modify best performing NN
            
    #     # Mutate and add new neural network
    #     modified_nn = nn_to_modify.mutate_and_create_copy(mutation_rate, modify_by_normal_dist)
    #     # Calculate the fitness
    #     try:
    #         nn_fitness, settings = get_fitness(wn, modified_nn, initial_pressures,
    #                                            episodes=episode_length, scaling=scaling)
    #     except:
    #         print('failed to execute sim, assigning failure')
    #         nn_fitness = -int(input_dim)
    #         settings = ['fail' for _ in range(output_dim)]
            
    #     modified_nn.define_fitness(nn_fitness)
    #     modified_nn.define_settings(settings)
    #     # Append new NN to list
    #     nn_population.append(modified_nn)
    #     nn_fitness_list.append(modified_nn.fitness)
    #     nn_settings_list.append(modified_nn.settings)
        
    #     # Determine solution that is the worst
    #     min_index = nn_fitness_list.index(min(nn_fitness_list))
    #     # Remove the worst solution
    #     nn_population.pop(min_index)
    #     nn_fitness_list.pop(min_index)
    #     nn_settings_list.pop(min_index)
        
        
    moving_average_reward_list = moving_average(best_performer, window=moving_window)
    iteration_num_move = [i*1 + 1*moving_window for i in range(len(moving_average_reward_list))]
    plot_scatter(iteration_num_move, moving_average_reward_list,
                 f'Best Performing NN Rewards over {moving_window} Iterations',
                 'Reward')
    plt.savefig(save_folder_path + 'learning_plot.png',
                dpi=600, bbox_inches='tight')
    plt.close()
    
    #%% Plot best solution
    # Get best NN and get the valve controls
    best_nn = heapq.nlargest(1, a_nn_population_dict, 
                                key=lambda k: a_nn_population_dict[k]['fitness'])
    best_nn_model = a_nn_population_dict[best_nn[0]]['nn']
    best_fitness = a_nn_population_dict[best_nn[0]]['fitness']
    # Get valve settings, should be the same, debugging
    valve_settings = a_nn_population_dict[best_nn[0]]['setting'][0]
    valve_settings_pass = list(best_nn_model.forward(initial_pressures)[0] * scaling)
    
    print('Finished running')
    print(f'Performance with valves open: {fitness_count_init}')
    print(f'Fitness according to list of best performing: {best_fitness}')
    
    wn.options.hydraulic.demand_model = 'PDD' # pressure driven analysis (better for leaky pipes)
    wn.options.time.duration = 0 # steady state, snapshot
    wn.options.hydraulic.headloss = headloss
    
    # Set all FCVs to some max limit indicating they are open
    for index, valve in enumerate(valves):
        valve_obj = wn.get_link(valve) # links are valves
        valve_obj.initial_setting = valve_settings[index] 
    
    # Simulation
    sim = wntr.sim.EpanetSimulator(wn)
    # sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim() # by default, this runs EPANET 2.2.0
    pressure = results.node['pressure']
    flow = results.link['flowrate']
    final_pressures = pressure.iloc[0] #pressures at 0
    final_pressures = final_pressures.apply(lambda x: max(x, 0)) # Set everything below 0 to 0
    # Remove new junctions added to be able to graph with structure
    valve_junction_list = [item.replace('FCV', 'V') for item in valves]
    final_pressures_plot = final_pressures.drop(valve_junction_list)
    
    fitness_count = 0
    for pressure in final_pressures:
        if 30 <= pressure <= 100:
            fitness_count += 1
    print(f'Best performance with valve control: {fitness_count}')
    print(f'Valve Settings: {valve_settings}')
    
    # Plot
    ax_p = wntr.graphics.plot_network(wn_plotting, node_attribute=final_pressures_plot,
                                      title='Post-Disaster Pressures with Valve Control',node_range=[0, 80],
                                      node_size=80,add_colorbar=True,node_colorbar_label='Pressure')
    plt.savefig(save_folder_path + 'post_disaster_with_valve_pressure_map.png',
                dpi=600, bbox_inches='tight')
    plt.close()
    # ax_f = wntr.graphics.plot_network(wn, link_attribute=intial_flow,title='Pre-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow')
    
    with open(save_folder_path + 'results.txt', 'w') as file:
        # Write data to the file
        file.write(f'Performance before disaster: {fitness_count_pre}\n')
        file.write(f'Performance with valves open: {fitness_count_init}\n')
        file.write(f'Performance with controlled valves: {best_fitness}\n')
        file.write(f'Valve Settings: {valve_settings}\n')