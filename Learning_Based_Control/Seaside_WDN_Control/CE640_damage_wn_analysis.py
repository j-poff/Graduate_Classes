# Import modules
import numpy as np
import pandas as pd
import geopandas as gpd # For reading in shapefiles : another way. 
import os # For managing directories and file paths if drive is mounted
import matplotlib.pyplot as plt  # for plot.
import wntr
import generate_epa_model as gepa
import copy
import pickle
import warnings
warnings.filterwarnings('ignore')
#%% Functions
def modify_inp_with_pipe_damage(df_pipeline_dmg, pipe_mapping_dict, wn, break_percent=0.20, 
                                leak_percent=0.03, close_pipes=True, break_time=0, fix_time=None):
    '''
    This function takes 
    
    Parameters
    ----------
    df_pipeline_dmg : pandas dataframe
        Dataframe from INCORE PipelineDamageRepairRate function.
    pipe_mapping_dict : dictionary
        Dictionary that maps pipe GUIDs from INCORE to pipe names in EPANET model.
    wn : WNTR WaterNetworkModel
        Water network object that will be modified to include leaks and breaks.
    break_percent : float, optional
        For break damage, percent of the pipe area that will be leaking. 
        See https://usepa.github.io/WNTR/hydraulics.html#leak-model. The default is 0.20.
    leak_percent : float, optional
        For leak damage, percent of the pipe area that will be leaking. 
        See https://usepa.github.io/WNTR/hydraulics.html#leak-model. The default is 0.03.
    close_pipes : boolean, optional
        If broken, close the pipes to flow. The default is True.
    break_time : float, optional
        The time during the simulation the leak/pipe closure will occur in seconds.
        The default is 0.
    fix_time : float, optional
        The time during the simulation the leak will stop in seconds.
        The default is None, meaning it will not stop.

    Returns
    -------
    Dictionary
        If return_leak_info = True, a dictionary that lists what pipes and nodes should have 
        leaks and breaks assigned
    '''

    # Create copy of pipeline damage dataframe
    df_pipeline_dmg_in = df_pipeline_dmg.copy()
    df_pipeline_dmg_in.set_index('guid',inplace=True)
    pipe_node_damage_dict = {}
    
    # Figure out how many breaks and leaks are estimated based on pipe length and rates calculated previously
    pgv_repairs_sum = df_pipeline_dmg_in['numpgvrpr'].sum()
    pgd_repairs_sum = df_pipeline_dmg_in['numpgdrpr'].sum()
    break_number = round(0.2 * pgv_repairs_sum + 0.8 *pgd_repairs_sum)
    leak_number = round(0.8 *pgv_repairs_sum + 0.2 * pgd_repairs_sum)
    print(f'Number of leaks: {leak_number}, Number of breaks: {break_number}')
    
    # Pick the broken pipes based on their breakrate
    df_pipeline_dmg_in['break_prob'] = df_pipeline_dmg_in['breakrate'] / df_pipeline_dmg_in['breakrate'].sum()
    broken_pipes_guids = np.random.choice(df_pipeline_dmg_in.index, size=break_number, replace=False, p=df_pipeline_dmg_in['break_prob'])
    pipe_node_damage_dict['broken_pipes'] = [pipe_mapping_dict[x] for x in list(broken_pipes_guids)]
    # Pick the pipes with leaks based on their leakrate, not including ones that are broken
    df_pipeline_dmg_in.loc[broken_pipes_guids, 'leakrate'] = 0 # Change leakrate for broken pipes to 0
    df_pipeline_dmg_in['leak_prob'] = df_pipeline_dmg_in['leakrate'] / df_pipeline_dmg_in['leakrate'].sum()
    leaky_pipes_guids = np.random.choice(df_pipeline_dmg_in.index, size=leak_number, replace=False, p=df_pipeline_dmg_in['leak_prob'])
    pipe_node_damage_dict['leaky_pipes'] = [pipe_mapping_dict[x] for x in list(leaky_pipes_guids)]
    
    ### Add leaks to 'broken' pipes, and disconnect (close) the pipe?
    nodes_with_leaks = []
    all_break_nodes_considered = []
    break_nodes_leak_area = []
    extra_breaks_to_add = 0
    for broken_pipe_guid in broken_pipes_guids:
        pipe_object = wn.get_link(pipe_mapping_dict[broken_pipe_guid])
        node1, node2, diameter = pipe_object.start_node.name, pipe_object.end_node.name, pipe_object.diameter # Get starting and ending nodes.
        all_break_nodes_considered.append(node1)
        all_break_nodes_considered.append(node2)
        tot_pipe_area = np.pi * diameter**2 / 4
        if node1 in nodes_with_leaks and node2 in nodes_with_leaks: # both pipes are already selected by some chance
            extra_breaks_to_add += 1
        else:
            if node1 in nodes_with_leaks: # pick node2
                leak_node = wn.get_node(node2) 
                nodes_with_leaks.append(node2)
                break_nodes_leak_area.append((node2,tot_pipe_area))
            else: # Choose node1
                leak_node = wn.get_node(node1)
                nodes_with_leaks.append(node1)
                break_nodes_leak_area.append((node1,tot_pipe_area))
            leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
            leak_node.add_leak(wn, area=tot_pipe_area*break_percent, start_time=break_time, end_time=fix_time)
            if close_pipes:
                #pipe_object.initial_status = 0 # try this?
                act = wntr.network.controls.ControlAction(pipe_object, 'status', 
                                                          wntr.network.LinkStatus.Closed)
                cond = wntr.network.controls.SimTimeCondition(wn, '=', break_time/3600)# hour in decimal
                ctrl = wntr.network.controls.Control(cond, act)
                wn.add_control('close pipe ' + pipe_mapping_dict[broken_pipe_guid], ctrl)
    # If extra breaks are necessary add them here
    if extra_breaks_to_add > 0:
        for _ in range(extra_breaks_to_add): # for as many additional breaks need to be added
            for break_node in all_break_nodes_considered:
                if break_node not in nodes_with_leaks: # That means we can add it
                    leak_node = wn.get_node(break_node)
                    nodes_with_leaks.append(break_node)
                    leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                    leak_node.add_leak(wn, area=tot_pipe_area*break_percent, start_time=break_time, end_time=fix_time)
                    if close_pipes:
                        pipe_object.initial_status = 0 # try this?
                    break
    pipe_node_damage_dict['broken_nodes'] = break_nodes_leak_area.copy() # get copy of just nodes that have broken
         
    ### Add leaks to the 'leaky' pipes
    all_leak_nodes_considered = []
    leak_nodes_leak_area = []
    extra_leaks_to_add = 0
    for leaky_pipe_guid in leaky_pipes_guids:
        pipe_object = wn.get_link(pipe_mapping_dict[leaky_pipe_guid])
        node1, node2, diameter = pipe_object.start_node.name, pipe_object.end_node.name, pipe_object.diameter # Get starting and ending nodes.
        all_leak_nodes_considered.append(node1)
        all_leak_nodes_considered.append(node2)
        tot_pipe_area = np.pi * diameter**2 / 4
        if node1 in nodes_with_leaks and node2 in nodes_with_leaks: # both pipes are already selected by some chance
            extra_leaks_to_add += 1
        else:
            if node1 in nodes_with_leaks: # pick node2
                leak_node = wn.get_node(node2) 
                nodes_with_leaks.append(node2)
                leak_nodes_leak_area.append((node2,tot_pipe_area))
            else: # Choose node1
                leak_node = wn.get_node(node1)
                nodes_with_leaks.append(node1)
                leak_nodes_leak_area.append((node1,tot_pipe_area))
            leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
            leak_node.add_leak(wn, area=tot_pipe_area*leak_percent, start_time=break_time, end_time=fix_time)
    # If extra leaks are necessary do some stuff here
    if extra_leaks_to_add > 0:
        for _ in range(extra_leaks_to_add): # for as many additional breaks need to be added
            for break_node in all_leak_nodes_considered:
                if break_node not in nodes_with_leaks: # That means we can add it
                    leak_node = wn.get_node(break_node)
                    nodes_with_leaks.append(break_node)
                    leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                    leak_node.add_leak(wn, area=tot_pipe_area*leak_percent, start_time=break_time, end_time=fix_time)
                    break
    pipe_node_damage_dict['leaky_nodes'] = leak_nodes_leak_area.copy()
    return pipe_node_damage_dict

    
def modify_inp_with_facility_damage(df_wterfclty_dmg, facility_mapping_dict, wn, break_time=0, fix_time=None):
    pump_damage_info = {'damaged_pumps':[],'damaged_tanks':[],'damaged_reservoir':[]}
    for index, row in df_wterfclty_dmg.iterrows():
        if row['haz_expose'] == 'yes':
            probabilities = row[['DS_0', 'DS_1', 'DS_2', 'DS_3','DS_4']].values
            damage_state = np.random.choice([0,1,2,3,4], p=list(probabilities))
            if damage_state > 2: # if damage state 3 or 4
                facility_epa_name = facility_mapping_dict[row['guid']] # Assuming dict maps guid with epa pump/tank/reservoir name as values
                pump = wn.get_link(facility_epa_name)
                pump.add_outage(wn, break_time, fix_time)
                print(f'Modified Pump: {facility_epa_name}')
                pump_damage_info['damaged_pumps'].append(facility_epa_name)
        #This only works for pumps, assuming reservoirs and tanks have all damage probabilities set to 0. Seaside is ok cause no tanks
    return pump_damage_info

def default_water_network_sim(wn,full_graphs=False, savepath=None,extra_name=""):
    n_size = 30
    l_width = 1.4
    ### Hydrualic simulation
    wn.options.time.duration = 2 * 3600 # 2 hours, steady state
    wn.options.hydraulic.headloss = 'H-W'
    wn.options.hydraulic.demand_model = 'PDD'
    # Conversion: 1 PSI to 0.7032496149020262 m
    wn.options.hydraulic.minimum_pressure = 0 #m / 0 PSI # 3.516m / 5 PSI. Pressure at which demand goes to 0  
    wn.options.hydraulic.required_pressure = 17.57 # m / 25 PSI # 24.614m / 35 PSI. Pressure at which demand starts to decrease due to insufficient pressure
    wn.options.hydraulic.pressure_exponent = 0.5 # default is 0.5, higher number means a sharper decrease in demand from required pressure to minimum pressure.
    # sim = wntr.sim.EpanetSimulator(wn_full_damaged)
    sim_damaged = wntr.sim.WNTRSimulator(wn)
    results_damaged = sim_damaged.run_sim() # by default, this runs EPANET 2.2.0
    try:
        leak_values = results_damaged.node['leak_demand'].iloc[1] * 15850.3
        damaged_pressure = results_damaged.node['pressure'].iloc[1] * 1.4219702063247 #pressures at 0
        damaged_flow = abs(results_damaged.link['flowrate'].iloc[1] * 15850.3) # flow at 0, convert to gpm
        damaged_velocity = abs(results_damaged.link['velocity'].iloc[1] * 3.28084) # velocity at 0, convert to ft/s
        damage_status = results_damaged.link['status'].iloc[1] 
    except IndexError:
        print('Water Network sim only ran for 1 time step and threw an error :/')
        return results_damaged, np.nan, np.nan, np.nan
    ### Graphing
    if savepath == None:
        wntr.graphics.plot_network(wn, node_attribute=damaged_pressure,node_range=[0,120],title='Pressures',node_size=n_size,add_colorbar=True,node_colorbar_label='Pressure (PSI)')
        wntr.graphics.plot_network(wn, link_attribute=damage_status,link_range=[0,1],title='Pipe Status',link_width=l_width,node_size=0,link_cmap=['red','blue'],add_colorbar=True,link_colorbar_label='Pipe Closed')
        if full_graphs:
            wntr.graphics.plot_network(wn, link_attribute=damaged_flow,title='Flows',link_width=l_width,node_size=0,add_colorbar=True,link_colorbar_label='Flow (GPM)')
            wntr.graphics.plot_network(wn, link_attribute=damaged_velocity,title='Velocities',link_width=l_width,node_size=0,add_colorbar=True,link_colorbar_label='Velocity (ft/s)')
            wntr.graphics.plot_network(wn, node_attribute=leak_values,node_range=[0,None],title='Nodal Leaks',node_size=n_size,add_colorbar=True,node_colorbar_label='Leak Demand (GPM)')    
    else:
        wntr.graphics.plot_network(wn, node_attribute=damaged_pressure,node_range=[0,120],title='Pressures',node_size=n_size,add_colorbar=True,node_colorbar_label='Pressure (PSI)')
        plt.savefig(os.path.join(savepath, f'{extra_name}pressure.png'),dpi=300, bbox_inches='tight')
        plt.close()
        wntr.graphics.plot_network(wn, link_attribute=damage_status,link_range=[0,1],title='Pipe Status',link_width=l_width,node_size=0,link_cmap=['red','blue'],add_colorbar=True,link_colorbar_label='Pipe Closed')
        plt.savefig(os.path.join(savepath, f'{extra_name}pipe_status.png'),dpi=300, bbox_inches='tight')
        plt.close()
        if full_graphs:
            wntr.graphics.plot_network(wn, link_attribute=damaged_flow,title='Flows',link_width=l_width,node_size=0,add_colorbar=True,link_colorbar_label='Flow (GPM)')
            plt.savefig(os.path.join(savepath, f'{extra_name}flow.png'),dpi=300, bbox_inches='tight')
            plt.close()
            wntr.graphics.plot_network(wn, link_attribute=damaged_velocity,title='Velocities',link_width=l_width,node_size=0,add_colorbar=True,link_colorbar_label='Velocity (ft/s)')
            plt.savefig(os.path.join(savepath, f'{extra_name}velocity.png'),dpi=300, bbox_inches='tight')
            plt.close()
            wntr.graphics.plot_network(wn, node_attribute=leak_values,node_range=[0,None],title='Nodal Leaks',node_size=n_size,add_colorbar=True,node_colorbar_label='Leak Demand (GPM)')
            plt.savefig(os.path.join(savepath, f'{extra_name}leak_demand.png'),dpi=300, bbox_inches='tight')
            plt.close()
        plt.close('all')
    # Save results
    if savepath != None:
        with open(os.path.join(savepath,f'{extra_name}sim_results.pkl'), 'wb') as file:
            pickle.dump(results_damaged, file)
    ### Demand not met analysis
    # Compare expected demand with delivered demand based on PDA analysis which reduces demand based on pressure
    expected_demand = wntr.metrics.expected_demand(wn)#.iloc[1]
    # expected_demand_gpm = expected_demand * 15850.3 * 60 * 24 / 80
    demand = results_damaged.node['demand']
    demand.drop(columns=['R-1'], inplace=True)
    # Fill in negative demand with 0
    demand = demand.map(lambda x: 0 if x < 0 else x)
    # wsa_nt = wntr.metrics.water_service_availability(expected_demand, demand) # ratio, weights every junction equally
    fulfilled_avg_demand_ratio = sum(list(demand.iloc[1])) / sum(list(expected_demand.iloc[1]))
    ### Other stats
    avg_pressure = sum(damaged_pressure)/len(damaged_pressure)
    leak_values.drop(columns=['R-1'], inplace=True)
    tot_leak_demand = sum(leak_values)
    return results_damaged, fulfilled_avg_demand_ratio, avg_pressure, tot_leak_demand

def apply_damage_to_wn(wn, pipe_node_damage_info, pump_damage_info,break_time=0,fix_time=None):
    # Apply breaks, disconnect pipes then apply large leaks
    for pipe_name in pipe_node_damage_info['broken_pipes']:
        pipe_object = wn.get_link(pipe_name)
        act = wntr.network.controls.ControlAction(pipe_object, 'status', 
                                                  wntr.network.LinkStatus.Closed)
        cond = wntr.network.controls.SimTimeCondition(wn, '=', break_time/3600)# hour in decimal
        ctrl = wntr.network.controls.Control(cond, act)
        wn.add_control('close pipe ' + pipe_name, ctrl)
    for broken_node_name_leak in pipe_node_damage_info['broken_nodes']: #(pipe_name, leak_area)
        leak_node = wn.get_node(broken_node_name_leak[0]) 
        leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
        # Need to figure out how to add total pipe area for each node's pipe here.
        leak_node.add_leak(wn, area=broken_node_name_leak[1], start_time=break_time, end_time=fix_time)
    # Apply leaks
    for leaky_node_name_leak in pipe_node_damage_info['leaky_nodes']: #(pipe_name, leak_area)
        leak_node = wn.get_node(leaky_node_name_leak[0]) 
        leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
        leak_node.add_leak(wn, area=leaky_node_name_leak[1], start_time=break_time, end_time=fix_time)
    # Apply damage to pumps
    for pump in pump_damage_info['damaged_pumps']:
        pump_obj = wn.get_link(pump)
        pump_obj.add_outage(wn, break_time,fix_time)
    return

#%% Metrics to capture for analysis:
'''
do 10 iterations of damage and then analyze with 3 different demand placements
METRICS
ratio of available demand to full demand (how much water is available from full average capacity)
average pressure across network
total leak demand
DIMENSIONS: return years
Demand placement:
    no evacuation
    bldg driven evacuation, tourist leave
    tsunami evacuation
'''

#%% Create default network and run simulation before disasters
# Set crs
crs = 'EPSG:32610' # UTM Zone 10N for Oregon, code EPSG:32610 in meters
# Load shapefiles
data_folder_path = os.path.join(os.getcwd(), 'data')
bldg_pop = gpd.read_file(os.path.join(data_folder_path, 'seaside_bldg_pop.geojson'))
water_pipelines = gpd.read_file(os.path.join(data_folder_path, 'Seaside_water_pipelines_wgs84.shp'))
water_nodes = gpd.read_file(os.path.join(data_folder_path, 'Seaside_wter_nodes.shp'))
list_dem_paths = [os.path.join(data_folder_path, 'seaside_elevation_raster.tif')]
# Add tourists in seasonal rentals to Seaside bldg population
bldg_pop_adj = gepa.seaside_bldg_pop_adjustment(bldg_pop, crs, tourist_population = 2000)
#  Create water network with adjusted bldg pop
# pre_wn = gepa.create_seaside_wn(bldg_pop_adj, water_pipelines, water_nodes, list_dem_paths, crs)
# Graph and get statistics
# pre_results, pre_dmnd_ratio, pre_avg_pressure, pre_leak_demand = default_water_network_sim(pre_wn,full_graphs=True, 
#                                                              savepath=os.path.join(os.getcwd(), 'output'),
#                                                              extra_name='original_')
# print(f'Percent of Average Demand Met: {round(pre_dmnd_ratio*100,1)}%')
# print(f'Average pressure: {round(pre_avg_pressure,1)} PSI')
# print(f'Leak Demand: {pre_leak_demand}')
# pre_wn.reset_initial_values() # reset simulation times
#%% Create mappings for damage translation

water_pipelines_guid = water_pipelines.set_index('guid',inplace=False)
pipe_mapping_dict = { guid_index: f'P-{row["Link_ID"]}' for guid_index, row in water_pipelines_guid.iterrows()} 
# Mapping for facilities  
facility_mapping_dict = {'2048ed28-395e-4521-8fc5-44322534592e' : 'R-1',
 '8d22fef3-71b6-4618-a565-955f4efe00bf' : 'TillaHPHSE',
 'cfe182a2-c39c-4734-bcd5-3d7cadab8aff' : 'PHouse',
 'd6ab5a29-1ca1-4096-a3c3-eb93b2178dfe' : 'RegalHPHSE'}

#%% Main loop through the system
results_dict = {}
rt = [100, 250, 500, 1000, 2500, 5000, 10000]
iterations = 10
savingfolder = r"E:\ce640_results"
for ret_prd in rt:
    print(f'Return Year: {ret_prd}________________')
    # Load appropiate damage dataframes
    path_to_data_folder = os.path.join(os.getcwd(), 'output', '{}yr' .format(ret_prd))
    df_pipeline_dmg =  pd.read_csv(os.path.join(path_to_data_folder,  'pipeline_eq_{}yr.csv' .format(ret_prd)))
    df_wterfclty_dmg =  pd.read_csv(os.path.join(path_to_data_folder,  'wterfclty_eq_{}yr.csv' .format(ret_prd)))
    df_bldg_dmg =   pd.read_csv(os.path.join(path_to_data_folder,  'buildings_cumulative_{}yr.csv' .format(ret_prd)))
    # Create dictionary and folders to store results
    demand_placements = ['no_evac', 'tsu_evac', 'bldg_evac']
    results_dict[ret_prd] = {evac:{'demand_ratio':[],'avg_pressure':[],'leak_demand':[]}
                             for evac in demand_placements}
    for evac in demand_placements:
        os.makedirs(os.path.join(savingfolder, f'{ret_prd}yr',evac),exist_ok=True)
#%% Translate pipe damage
    # Iterate x times
    
    for x in range(iterations):
        print(f'______________Iteration {x+1}_____________')
        wn = gepa.create_seaside_wn(bldg_pop_adj, water_pipelines, water_nodes, list_dem_paths, crs)
        pipe_node_damage_info = modify_inp_with_pipe_damage(df_pipeline_dmg, 
                                                                         pipe_mapping_dict, 
                                                                         wn, break_percent=0.4,
                                                                         leak_percent=0.1,
                                                                         break_time=1*3600,
                                                                         fix_time=2*3600,
                                                                         close_pipes=True)
        pump_damage_info = modify_inp_with_facility_damage(df_wterfclty_dmg, facility_mapping_dict, wn)
    
    #%% See how networks performs if no one moves? NO EVAC
        save_path = os.path.join(savingfolder, f'{ret_prd}yr','no_evac')
        results, dmnd_ratio, avg_pressure, leak_demand = default_water_network_sim(wn,full_graphs=True, 
                                                                     savepath=save_path,
                                                                     extra_name=f'iter{x+1}_')
        results_dict[ret_prd]['no_evac']['demand_ratio'].append(dmnd_ratio)
        results_dict[ret_prd]['no_evac']['avg_pressure'].append(avg_pressure)
        results_dict[ret_prd]['no_evac']['leak_demand'].append(leak_demand)
    
    #%% Calculate population shift due to tsunami evacuation and get new demand values
        plot = False
    
        ### OPTION 1: TSUNAMI EVACUATION. All residents/tourists flee to designated areas
        
        # Load evacuation zone polygons and building shpfile     
        evac_zone_polygons = gpd.read_file(os.path.join(data_folder_path, 'Evacuation_Zones.shp'))
        evac_zone_polygons.to_crs(crs,inplace=True)
        evac_zone_polygons.drop(columns=['id'], inplace=True)
        bldg_pop_adj.to_crs(crs,inplace=True)
        if plot:
            ax = evac_zone_polygons.plot(color='red', alpha=0.5, figsize=(10, 10))
            bldg_pop_adj.plot(ax=ax, color='blue', alpha=0.5)
            plt.legend(['Evac Zones', 'Buildings'])
            
        # Spatial Join to give buildings an evacuation zone
        buildings_within_evac_zones = gpd.sjoin(bldg_pop_adj, evac_zone_polygons, how='left', op='within')
        buildings_within_evac_zones['ezac_zone'].fillna(0, inplace=True) # 0 means no evac
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Adjust figure size as needed
            buildings_within_evac_zones.plot(column='ezac_zone', cmap='viridis', ax=ax, legend=True)
            plt.title('Building Evac Zone. 0 = No evac')
            
        # Move populations to designated evac zones
        evac_building_dict = {4:"e0226f6d-9c99-4017-a7a7-e59f2033effc", # elementry school
                              3: "7920553d-4639-4bbf-9eaf-ebf94069655d", # hospital
                              2:"56a7148d-7a32-47ae-b61e-4b0feca99578", # some house..?
                              1:"6b692e18-7395-47bc-b7dc-12fafcff9e6b", # Nourish those in need, charity organization
                              5: "3be061e5-bd24-45d6-8348-2876c1455a10" # Whalers Point Condo Complex
                              }
        for bldg_guid, row in buildings_within_evac_zones.iterrows():
            if row['ezac_zone'] in evac_building_dict: # not 0
                # Get population of bldg to be evacuated, and safe house
                evac_bldg_pop = row['numprec']
                safe_bldg_pop = buildings_within_evac_zones.loc[evac_building_dict[row['ezac_zone']], 'numprec']
                # Add population to existing population at safe house
                buildings_within_evac_zones.loc[evac_building_dict[row['ezac_zone']], 'numprec'] = evac_bldg_pop + safe_bldg_pop
                # Set population of evacuated bldg to 0
                buildings_within_evac_zones.loc[bldg_guid, 'numprec'] = 0
                
        tsunami_evac_wn = gepa.create_seaside_wn(buildings_within_evac_zones, water_pipelines, water_nodes, list_dem_paths, crs)
        
        # Apply the same damage to the water network
        apply_damage_to_wn(tsunami_evac_wn, pipe_node_damage_info, pump_damage_info,break_time=1*3600,fix_time=2*3600)
        # Power up all pumps, some might be turned off. small pumps typically 250 - 1100W, bigger can be 2000-5000W
        tsunami_evac_wn.get_link('RegalHPHSE').power = 1000 #regal 0.005 HP, 3.677 W
        tsunami_evac_wn.get_link('PHouse').power = 1000 #phouse 0.3 HP, 220.65 W
        tsunami_evac_wn.get_link('TillaHPHSE').power = 1000# tillamook 0.15 HP, 110.325 W
        # Do simulation analysis
        save_path = os.path.join(savingfolder, f'{ret_prd}yr','tsu_evac')
        results, dmnd_ratio, avg_pressure, leak_demand = default_water_network_sim(tsunami_evac_wn,full_graphs=True, 
                                                                     savepath=save_path,
                                                                     extra_name=f'iter{x+1}_')
        results_dict[ret_prd]['tsu_evac']['demand_ratio'].append(dmnd_ratio)
        results_dict[ret_prd]['tsu_evac']['avg_pressure'].append(avg_pressure)
        results_dict[ret_prd]['tsu_evac']['leak_demand'].append(leak_demand)
        
    #%% Calculate population shift due to bldg dmg and get new demand values    
        ### OPTION 2: Post tsunami, some residents can return home (if home not damaged), 
        ###           tourists leave city, others stay in shelters (schools?)
        if df_bldg_dmg.index.name != 'guid':
            df_bldg_dmg.set_index('guid',inplace=True)
        dmg_leave_prob=[0,.3,.8,1]
        evac_centers = ['e0226f6d-9c99-4017-a7a7-e59f2033effc','7920553d-4639-4bbf-9eaf-ebf94069655d']
        bldg_pop_adj_move = bldg_pop_adj.copy() # don't overwrite original copy
        for guid, row in bldg_pop_adj_move.iterrows():
            if row['numprec'] > 0 and guid not in evac_centers: # if not empty and not an evac center
                # Determine damage state
                dmg_df_row = df_bldg_dmg.loc[guid]
                probabilities = dmg_df_row[['DS_0', 'DS_1', 'DS_2', 'DS_3']].values
                damage_state = np.random.choice([0,1,2,3], p=list(probabilities))
                leaving_prob = dmg_leave_prob[damage_state]
                # Simulate if leaving or staying
                if np.random.uniform(0, 1) < leaving_prob:
                    # Left!
                    if row['landuse'] in ['losr','hosr']:
                        # tourist, leave town
                        bldg_pop_adj_move.loc[guid,'numprec'] = 0
                    else: # random choice between hospital and school
                        evac_choice = np.random.choice(evac_centers)
                        current_evac_pop = bldg_pop_adj_move.loc[evac_choice,'numprec']
                        bldg_pop_adj_move.loc[evac_choice,'numprec'] = current_evac_pop + bldg_pop_adj_move.loc[guid,'numprec']
                        bldg_pop_adj_move.loc[guid,'numprec'] = 0
                        
        # Get new water network
        bldg_evac_wn = gepa.create_seaside_wn(buildings_within_evac_zones, water_pipelines, water_nodes, list_dem_paths, crs)
        # obj = wntr.epanet.InpFile()
        # obj.write(f'tsunami_evac_seaside.inp',bldg_evac_wn)
        
        # Apply the same damage to the water network
        apply_damage_to_wn(bldg_evac_wn, pipe_node_damage_info, pump_damage_info,break_time=1*3600,fix_time=2*3600)
        # Power up all pumps, some might be turned off. small pumps typically 250 - 1100W, bigger can be 2000-5000W
        #bldg_evac_wn.get_link('RegalHPHSE').power = 1000 #regal 0.005 HP, 3.677 W
        bldg_evac_wn.get_link('PHouse').power = 1000 #phouse 0.3 HP, 220.65 W
        #bldg_evac_wn.get_link('TillaHPHSE').power = 1000# tillamook 0.15 HP, 110.325 W
        # Do simulation analysis
        save_path = os.path.join(savingfolder, f'{ret_prd}yr','bldg_evac')
        results, dmnd_ratio, avg_pressure, leak_demand = default_water_network_sim(bldg_evac_wn,full_graphs=True, 
                                                                     savepath=save_path,
                                                                     extra_name=f'iter{x+1}_')
        results_dict[ret_prd]['bldg_evac']['demand_ratio'].append(dmnd_ratio)
        results_dict[ret_prd]['bldg_evac']['avg_pressure'].append(avg_pressure)
        results_dict[ret_prd]['bldg_evac']['leak_demand'].append(leak_demand)
    # Finished iterations
    save_path = os.path.join(savingfolder, f'{ret_prd}yr')
    with open(os.path.join(save_path,'results.pkl'), 'wb') as file:
        pickle.dump(results_dict[ret_prd], file)
    
    
