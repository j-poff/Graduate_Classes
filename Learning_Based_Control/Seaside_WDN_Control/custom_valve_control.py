
import os
import matplotlib.pyplot as plt  
import wntr

import warnings
import json

#%% Functions
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

pipe_pair_for_valve = [ [772,785],#open
                       [491,493],#open
                       [684,810], # close
                       [39,38],
                       [452,451],
                       [297,425],
                       [428,426],
                       [139,86],
                       [1,5], #close
                       [55,53],
                       [819,651],#close
                       [330,346], #close
                       [440,441],#close
                       [508,526], #close
                       [510,633],#close
                       [786,830]
    ]

warnings.filterwarnings("ignore") # ignore all warnings for low pressure simulations   
    
#%% set variables
return_values = [100,250,500,1000,2500,5000,10000]
# return_values = [1000]
for rt_val in return_values:
    # rt_val = 1000 # 100,250,500,1000,2500,5000,10000 #FIXME
    scaling = 500 # Max flow should be what? GPM
    headloss = 'D-W'
    iterations = 'Manual_Control'
    
    os.makedirs(f'results/{rt_val}yr', exist_ok=True)
    os.makedirs(f'results/{rt_val}yr/{iterations}', exist_ok=True)
    save_folder_path = f'results/{rt_val}yr/{iterations}/'
    #%% Pre-Disaster
    wn_pre_disaster = wntr.network.WaterNetworkModel(os.getcwd() + '/src/Seaside_dummy_model.inp') #load water network
    
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
    #%% Post-Disaster
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
    
#%%
    # Set all FCVs to some user specified control
    # custom_valve_control = [1,1,0,1,
    #                         1,1,1,1,
    #                         0,1,0,0,
    #                         0,0,0,1]
    
    custom_valve_control = [1,1,0,1,
                            1,1,1,0,
                            0,1,0,0,
                            0,0,0,1]

    # import random
    # for _ in range(100):
        # Generate a list of 16 random 0s and 1s
        # custom_valve_control = [random.choice([0, 1]) for _ in range(16)]
        
    for index, valve in enumerate(valves):
        valve_obj = wn.get_link(valve) # links are valves
        valve_obj.initial_setting = custom_valve_control[index] * scaling
        
    # Initial simulation
    sim = wntr.sim.EpanetSimulator(wn)
    # sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim() # by default, this runs EPANET 2.2.0
    pressure = results.node['pressure']
    
    initial_pressures = pressure.iloc[0] #pressures at 0
    initial_pressures = initial_pressures.apply(lambda x: max(x, 0)) # Set everything below 0 to 0
    # Remove new junctions added to be able to graph with structure
    valve_junction_list = [item.replace('FCV', 'V') for item in valves]
    initial_pressures_plot = initial_pressures.drop(valve_junction_list)
    
    # Initial Plot
    ax_p = wntr.graphics.plot_network(wn_plotting, node_attribute=initial_pressures_plot,
                                      title='Post-Disaster Pressures with Manual Valve Control',node_range=[0, 80],
                                      node_size=80,add_colorbar=True,node_colorbar_label='Pressure, PSI')
    plt.savefig(save_folder_path + 'post_disaster_manual_control_valve_pressure_map.png',
                dpi=600, bbox_inches='tight')
    plt.close()
    # ax_f = wntr.graphics.plot_network(wn, link_attribute=intial_flow,title='Pre-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow')
    
    fitness_count_init = 0
    for pressure in initial_pressures:
        if 30 <= pressure <= 100:
            fitness_count_init += 1
    
    with open(save_folder_path + 'results.txt', 'w') as file:
        # Write data to the file
        file.write(f'Performance before disaster: {fitness_count_pre}\n')
        file.write(f'Performance with manual valve control, close edges: {fitness_count_init}\n')