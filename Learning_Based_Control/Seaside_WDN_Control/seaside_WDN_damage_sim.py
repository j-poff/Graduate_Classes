#pyincore installation guide https://incore.ncsa.illinois.edu/doc/incore/pyincore/install_pyincore.html
from pyincore import IncoreClient, Dataset, DataService, HazardService, FragilityService
from pyincore import FragilityCurveSet, MappingSet
# from pyincore_viz.geoutil import GeoUtil as geoviz
# from pyincore_viz.plotutil import PlotUtil as plotviz
from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.pipelinedamage import PipelineDamage
from pyincore.analyses.waterfacilitydamage import WaterFacilityDamage
from pyincore.analyses.pipelinedamagerepairrate import PipelineDamageRepairRate
from pyincore.analyses.housingunitallocation import HousingUnitAllocation
import pandas as pd
import geopandas as gpd # For reading in shapefiles : another way. 
import numpy as np
import os # For managing directories and file paths if drive is mounted
import matplotlib.pyplot as plt  # for plot.
import wntr
import math
#%%
load_results = False
return_val = 10000
#%%
client = IncoreClient() # To get account, go to https://incore.ncsa.illinois.edu/doc/incore/account.html
data_service = DataService(client)
# Reading the water infrastrcuture: water pipes, a water treatment plant, and three water pumping stations. 
wter_pipe_dataset_id = "60e72f9fd3c92a78c89636c7"        # defining water pipes (GIS point layer)
wter_pipe_dataset = Dataset.from_data_service(wter_pipe_dataset_id, data_service)
pipeline_df = wter_pipe_dataset.get_dataframe_from_shapefile()

#%%
if not load_results:
    #geoviz.plot_map(wter_pipe_dataset, column=None, category=False)
    
    wter_fclty_dataset_id = "60e5e91960b3f41243faa3b2"        # defining water facilities (GIS point layer)
    wter_fclty_dataset = Dataset.from_data_service(wter_fclty_dataset_id, data_service)
    #geoviz.plot_map(wter_fclty_dataset, column='utilfcltyc', category=True)
    #%%
    # Define hazard models: earthquake and tsunami
    
    rt = [100, 250, 500, 1000, 2500, 5000, 10000]
    
    hazard_type = "earthquake"
    eq_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
                      250: "5dfa41aab9219c934b64d4b2",
                      500: "5dfa4300b9219c934b64d4d0",
                      1000: "5dfa3e36b9219c934b64c231",
                      2500: "5dfa4417b9219c934b64d4d3", 
                      5000: "5dfbca0cb9219c101fd8a58d",
                     10000: "5dfa51bfb9219c934b68e6c2"}
    
    #hazard_type = "tsunami"
    tsu_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
                      250: "5df910abb9219cd00cf5f0a5",
                      500: "5df90e07b9219cd00ce971e7",
                      1000: "5df90137b9219cd00cb774ec",
                      2500: "5df90761b9219cd00ccff258",
                      5000: "5df90871b9219cd00ccff273",
                      10000: "5d27b986b9219c3c55ad37d0"}
    #%%
    # Calculate the water infrastructure damage due to earthquake!
    
    #-- Pipe damage
    pipeline_eq_dmg_result_list = []  # place holder to saving earthquake pipe damage result iteration
    
    ### Conversion from .inp file to shapefile must be done manually to geolocate? 
    ### Pipes should share names (or a mapping dictionary) between EPAnet and the GUID of the shapefile.
    pipe_dataset_id = "60e72f9fd3c92a78c89636c7"             # defining pipe dataset (GIS layer);
    mapping_id = "5b47c227337d4a38464efea8"                  # specifiying mapping id from fragilites to pipe types
    fragility_service = FragilityService(client)             # loading fragility mapping
    mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
    
    #for rt_val in rt:                                        # loop through recurrence interval
    rt_val = return_val                                             # Set recurrence interval
    
    pipeline_dmg = PipelineDamageRepairRate(client)                    # initializing pyincore
    pipeline_dmg.load_remote_input_dataset("pipeline", pipe_dataset_id) # loading in the above id
    mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
    pipeline_dmg.set_input_dataset("dfr3_mapping_set", mapping_set)
    
    result_name = 'pipeline_eq_' + str(rt_val) + 'yr_pipeline_result' # defining output name
    
    pipeline_dmg.set_parameter("hazard_type", "earthquake")  # defining hazard type (e.g. earthquake vs. tsunami)
    hazard_id = eq_hazard_dict[rt_val]                   # specifying hazard id for specific recurrence interval
    pipeline_dmg.set_parameter("hazard_id", hazard_id)       # loading above into pyincore
    pipeline_dmg.set_parameter("fragility_key", "pgv")
    pipeline_dmg.set_parameter("num_cpu", 6)                 # number of CPUs to use for parallel processing
    pipeline_dmg.set_parameter("result_name", result_name)   # specifying output name in pyincore
    #%%
    pipeline_dmg.run_analysis()                              # running the analysis with the above parameters: it gives you 
    pipeline_eq_dmg_result_list.append(pipeline_dmg.get_output_dataset('result'))
    #%%

    # Convert dataset to Pandas DataFrame:
    df_pipeline_dmg_250 = pd.read_csv(f'pipeline_eq_{rt_val}yr_pipeline_result.csv')
    
    # Plot pipeline damage for 250-year return period. pyincore returns failure probs for EQ.
    #df_pipeline_dmg_250.head() 
    
    #%% 
    ### Pumps, tanks, reservoirs, valves should share names. Damage to valves??
    #-- Facility damage (pumps, tanks, reservoirs?)
    wterfclty_eq_dmg_result_list = []  # place holder to saving earthquake wterfclty damage result iteration
    fclty_dataset_id = "60e5e91960b3f41243faa3b2"             # defining Facility dataset (GIS layer);
    mapping_id = "5d39e010b9219cc18bd0b0b6"                  # specifiying mapping id from fragilites to pipe types
    fragility_service = FragilityService(client)             # loading fragility mapping
    mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
    
    #for rt_val in rt:                                        # loop through recurrence interval
    rt_val = return_val
    
    wterfclty_dmg = WaterFacilityDamage(client)                    # initializing pyincore
    wterfclty_dmg.load_remote_input_dataset("water_facilities", fclty_dataset_id) # loading in the above id
    mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
    wterfclty_dmg.set_input_dataset("dfr3_mapping_set", mapping_set)
    
    result_name = 'wterfclty_eq_' + str(rt_val) + 'yr_wterfclty_result' # defining output name
    
    wterfclty_dmg.set_parameter("hazard_type", "earthquake")  # defining hazard type (e.g. earthquake vs. tsunami)
    hazard_id = eq_hazard_dict[rt_val]                   # specifying hazard id for specific recurrence interval
    wterfclty_dmg.set_parameter("hazard_id", hazard_id)       # loading above into pyincore
    wterfclty_dmg.set_parameter("fragility_key", "pga")
    wterfclty_dmg.set_parameter("num_cpu", 6)                 # number of CPUs to use for parallel processing
    wterfclty_dmg.set_parameter("result_name", result_name)   # specifying output name in pyincore
    #%%
    wterfclty_dmg.run_analysis()                              # running the analysis with the above parameters: it gives you 
    wterfclty_eq_dmg_result_list.append(wterfclty_dmg.get_output_dataset('result'))
    
    ###
    # Convert dataset to Pandas DataFrame: example of 2500-yr return period
    df_wterfclty_dmg_250 = pd.read_csv(f'wterfclty_eq_{rt_val}yr_wterfclty_result.csv')
    
    # Plot wterfclty damage for 250-year return period. # LS_0 is the probability of exceeding each damage state; DS_0 is the probability of being in damage state! 
    #df_wterfclty_dmg_250.head() 
else:
    df_pipeline_dmg_250 = pd.read_csv(f'damage_data/pipeline_eq_{return_val}yr_pipeline_result.csv')
    df_wterfclty_dmg_250 = pd.read_csv(f'damage_data/wterfclty_eq_{return_val}yr_wterfclty_result.csv')
#%%
def modify_inp_with_pipe_damage(df_pipeline_dmg, pipe_mapping_dict, wn, pipeline_df):
    
    # Function: Modify pipe network via pipeline damage
    did_anything = 0
    round_if_higher_or_equal_to_decimal = 0.8
    def custom_round(num,threshold):
        decimal_part = num - int(num)
        if decimal_part >=threshold:
            return int(num) + 1
        else:
            return int(num)
    if pipeline_df.index.name != 'guid':
        pipeline_df.set_index('guid',inplace=True)   
    wn_dict = wn.to_dict()
    break_time = 0
    fail_percent = 0.50
    break_percent = 0.20
    leak_percent = 0.03
    for index, row in df_pipeline_dmg.iterrows():
        random_float = np.random.random()
        
        assert row['guid'] in pipeline_df.index, 'Cannot find guid match from shp file'
        pipe_length = pipeline_df.loc[row['guid'],'length_km']
        pipe_epa_name = pipe_mapping_dict[row['guid']]['epa_pipes'] # Assuming dict maps guid with epa pipe name as values
        # Check if pipe totally fails
        if random_float < row['failprob']: # if below probability, it breaks!
            ##PIPE BREAK
            for pipe in pipe_epa_name: # Break all the epa pipes associated with the shp pipe
                # First pipe, add 50% area leak
                if pipe == pipe_epa_name[0]:
                    pipe_object = wn.get_link(pipe)
                    node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                    tot_pipe_area = np.pi / 4 * diameter**2
                    leak_node = wn.get_node(node1) # get starting node of leaky pipe
                    leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                    leak_node.add_leak(wn, area=tot_pipe_area*break_percent, start_time=break_time, end_time=None)
                    did_anything +=1
                # Last pipe, assuming there is more than one add 50% area leak on the end
                if len(pipe_epa_name) != 1 and pipe == pipe_epa_name[len(pipe_epa_name)-1]:
                    try:
                        pipe_object = wn.get_link(pipe)
                    except KeyError:
                        print(pipe_object)
                        print(row['guid'])
                    node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                    tot_pipe_area = np.pi / 4 * diameter**2
                    leak_node = wn.get_node(node2) # get starting node of leaky pipe
                    leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                    leak_node.add_leak(wn, area=tot_pipe_area*break_percent, start_time=break_time, end_time=None)
                    did_anything +=1
                # If only one pipe, add leak on other end
                if len(pipe_epa_name) == 1:
                    pipe_object = wn.get_link(pipe)
                    node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                    tot_pipe_area = np.pi / 4 * diameter**2
                    leak_node = wn.get_node(node2) # get starting node of leaky pipe
                    leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                    leak_node.add_leak(wn, area=tot_pipe_area*break_percent, start_time=break_time, end_time=None)
                    did_anything +=1
                # After adding leaks to nodes, close all pipes
                try:
                    pipe_object = wn.get_link(pipe)
                    pipe_object = wn.get_link(pipe)
                    pipe_object.initial_status = pipe_object.initial_status.Closed
                except KeyError:
                    print('Failed to close following pipes upon failure, not found in inp model?')
                    print(pipe_object)
                    print(row['guid'])
        else:
            # Calculate how many leaks and breaks
            num_of_leaks = custom_round(row['leakrate'] * pipe_length,round_if_higher_or_equal_to_decimal)
            num_of_breaks = custom_round(row['breakrate'] * pipe_length,round_if_higher_or_equal_to_decimal)
            num_epa_pipes = len(pipe_epa_name)
            pipes_messed_with = 0
            for i in range(num_of_breaks): # start assigning breaks
                if pipes_messed_with >= num_epa_pipes: #we've run out of pipes to mess with
                    break #exit for loop
                pipe_object = wn.get_link(pipe_epa_name[i])
                node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                tot_pipe_area = np.pi / 4 * diameter**2
                # new_leak_node = str(index) + '_leak_node' # create unique name for leak node
                # new_pipe_names = wntr.morph.split_pipe(wn, node1, node2, new_leak_node) #option to split pipe and add node for leak, or just add leak to a node?
                # new_pipe_names[0], new_pipe_names[1] = str(pipe_epa_name) + "_A", str(pipe_epa_name) + "_B" #need to adjust whole dictionary, not worth it....
                leak_node = wn.get_node(node1) # get starting node of leaky pipe
                leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                leak_node.add_leak(wn, area=tot_pipe_area*break_percent, start_time=break_time, end_time=None) # Add leak to starting node of pipe
                pipes_messed_with += 1
                did_anything +=1
            for i in range(num_of_breaks, num_of_breaks + num_of_leaks): #start assingning leaks where you left off
                if pipes_messed_with >= num_epa_pipes: #we've run out of pipes to mess with
                    break #exit for loop
                pipe_object = wn.get_link(pipe_epa_name[i])
                node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                tot_pipe_area = np.pi / 4 * diameter**2
                # new_leak_node = str(index) + '_leak_node' # create unique name for leak node
                # new_pipe_names = wntr.morph.split_pipe(wn, node1, node2, new_leak_node) #option to split pipe and add node for leak, or just add leak to a node?
                # new_pipe_names[0], new_pipe_names[1] = str(pipe_epa_name) + "_A", str(pipe_epa_name) + "_B" #need to adjust whole dictionary, not worth it....
                leak_node = wn.get_node(node1) # get starting node of leaky pipe
                leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                leak_node.add_leak(wn, area=tot_pipe_area*leak_percent, start_time=break_time, end_time=None) # Add leak to starting node of pipe
                pipes_messed_with += 1
                did_anything +=1
                
            # There's a chance we did not get through adding all the breaks and leaks dictated. Add those now.
            breaks_left = num_of_breaks - pipes_messed_with # how many breaks we didn't get to
            leaks_left = (num_of_breaks + num_of_leaks) - pipes_messed_with # how many leaks we didn't get to
            if breaks_left > 0: # add more breaks if not 0
                area_percentage_multiplier = math.ceil((breaks_left / num_epa_pipes)) + 1
                big_break_percent = break_percent * area_percentage_multiplier
                if big_break_percent > 1:
                    big_break_percent = 1
                if breaks_left % num_epa_pipes != 0:
                    for v in range(breaks_left % num_epa_pipes):
                        pipe_object = wn.get_link(pipe_epa_name[v])
                        node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                        tot_pipe_area = np.pi / 4 * diameter**2
                        leak_node = wn.get_node(node1) # get starting node of leaky pipe
                        leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                        leak_node.add_leak(wn, area=tot_pipe_area*big_break_percent, start_time=break_time, end_time=None) # Add leak to starting node of pipe
                        did_anything +=1
                else:
                    for v in range(num_epa_pipes):
                        pipe_object = wn.get_link(pipe_epa_name[v])
                        node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                        tot_pipe_area = np.pi / 4 * diameter**2
                        leak_node = wn.get_node(node1) # get starting node of leaky pipe
                        leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                        leak_node.add_leak(wn, area=tot_pipe_area*big_break_percent, start_time=break_time, end_time=None) # Add leak to starting node of pipe
                        did_anything +=1
            elif leaks_left > 0: # Add additional leaks, but ignore leaks if added more breaks, doesn't add much
                area_percentage_multiplier = math.ceil((leaks_left / num_epa_pipes)) + 1
                big_leak_percent = leak_percent * area_percentage_multiplier
                if big_leak_percent > 1:
                    big_leak_percent = 1
                if breaks_left % num_epa_pipes != 0:
                    for v in range(leaks_left % num_epa_pipes):
                        pipe_object = wn.get_link(pipe_epa_name[v])
                        node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                        tot_pipe_area = np.pi / 4 * diameter**2
                        leak_node = wn.get_node(node1) # get starting node of leaky pipe
                        leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                        leak_node.add_leak(wn, area=tot_pipe_area*big_leak_percent, start_time=break_time, end_time=None) # Add leak to starting node of pipe
                        did_anything +=1
                else:
                    for v in range(num_epa_pipes):
                        pipe_object = wn.get_link(pipe_epa_name[v])
                        node1, node2, diameter = pipe_object.start_node, pipe_object.end_node, pipe_object.diameter # Get starting and ending nodes.
                        tot_pipe_area = np.pi / 4 * diameter**2
                        leak_node = wn.get_node(node1) # get starting node of leaky pipe
                        leak_node.remove_leak(wn) # Remove any existing leak settings to add new ones
                        leak_node.add_leak(wn, area=tot_pipe_area*big_leak_percent, start_time=break_time, end_time=None) # Add leak to starting node of pipe
                        did_anything +=1
    print(f'leaks added {did_anything}')
    return wn
        
def modify_inp_with_facility_damage(df_wterfclty_dmg, facility_mapping_dict, wn):
    
    for index, row in df_wterfclty_dmg.iterrows():
        random_float = np.random.random()
        facility_epa_name = facility_mapping_dict[row['guid']] # Assuming dict maps guid with epa pump/tank/reservoir name as values
        if row['haz_expose'] == 'yes':
            if random_float > row['LS_2']: # if damage exceeds LS0 and LS1, turn pump off.
                pump = wn.get_link(facility_epa_name)
                pump.add_outage(wn, 0)
                print(f'Modified Pump: {facility_epa_name}')
        #This only works for pumps, assuming reservoirs and tanks have all damage probabilities set to 0. Seaside is ok cause no tanks
    return wn

#%% Creating dictionaries for use in code

import json
# Load pipe to shapefile GUID
with open('src/epa_pipe_to_guid_dict.json') as f:
    pipe_to_shp_guid = json.load(f)
# create shp_guid dictionary with epa pipes names, bldgs, and epa nodes.
shp_guid_to_info = {}
for key, value in pipe_to_shp_guid.items():
    if value not in shp_guid_to_info:
        shp_guid_to_info[value] = {'epa_pipes':[key]}
    else:
        shp_guid_to_info[value]['epa_pipes'].append(key)
        
# Add buildings to shp_guid dictionary
building2water = pd.read_csv(r"C:\Users\poffja\Box\Jason_Poff\INCORE_WNTR\Data\Seaside_Data\FreshWaterNetwork\2021-07_Water_Network\bldgs2wter_Seaside.csv")
for index, row in building2water.iterrows():
    edge_guid = row['edge_guid']
    bldg_guid = row['bldg_guid']
    if edge_guid in shp_guid_to_info.keys(): # make sure it's in the keys, just for bugs. shouldn't matter
        if 'bldgs' in shp_guid_to_info[edge_guid].keys(): # already created 'bldgs'
            shp_guid_to_info[edge_guid]['bldgs'].append(bldg_guid)
        else:
            shp_guid_to_info[edge_guid]['bldgs'] = [bldg_guid]
for key in shp_guid_to_info.keys(): # Add bldgs keys pipes with no bldgs assigned.
    if 'bldgs' not in shp_guid_to_info[key].keys():
        shp_guid_to_info[key]['bldgs'] = []
        
# Add epa nodes to dictionary.
wn = wntr.network.WaterNetworkModel(r"C:\Users\poffja\Box\Jason_Poff\INCORE_WNTR\Data\Seaside_dummy_model.inp") #load water network
wn.options.hydraulic.demand_model = 'PDD'
# wn = wntr.morph.skel.skeletonize(wn, 8*.0254, branch_trim=True, 
#                             series_pipe_merge=True, parallel_pipe_merge=True, 
#                             max_cycles=None, use_epanet=True, return_map=False, 
#                             return_copy=False)

wn_dict = wn.to_dict()
for pipe_guid in shp_guid_to_info.keys():
    connected_epa_nodes = []
    list_of_epa_pipes = shp_guid_to_info[pipe_guid]['epa_pipes']
    for epa_pipe in list_of_epa_pipes:
        str_node = wn_dict['links'][int(epa_pipe)]['start_node_name']
        end_node = wn_dict['links'][int(epa_pipe)]['end_node_name']
        if str_node not in connected_epa_nodes:
            connected_epa_nodes.append(str_node)
        if end_node not in connected_epa_nodes:
            connected_epa_nodes.append(end_node)
    shp_guid_to_info[pipe_guid]['epa_nodes'] = connected_epa_nodes

bldg_guid_demand_node_dict = {}
for shp_pipe in shp_guid_to_info.keys():
    num_bldgs = len(shp_guid_to_info[shp_pipe]['bldgs']) #number of buildings
    num_epa_nodes = len(shp_guid_to_info[shp_pipe]['epa_nodes'])
    counter = 0
    for bldg in shp_guid_to_info[shp_pipe]['bldgs']:
        if counter <= num_epa_nodes: # reset index when you've gone through all pipes
            counter = 0
        bldg_guid_demand_node_dict[bldg] = shp_guid_to_info[shp_pipe]['epa_nodes'][counter]
        counter += 1 # next pipe

facility_mapping_dict = {'2048ed28-395e-4521-8fc5-44322534592e' : 'R-1',
 '8d22fef3-71b6-4618-a565-955f4efe00bf' : 'TillaHPHSE',
 'cfe182a2-c39c-4734-bcd5-3d7cadab8aff' : 'PHouse',
 'd6ab5a29-1ca1-4096-a3c3-eb93b2178dfe' : 'RegalHPHSE'}

#%%
base_demands = []
for node_name in wn.nodes:
    node_obj = wn.get_node(node_name)
    if node_obj.node_type == 'Junction':
        base_demands.append(node_obj.base_demand)
#%%
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim() # by default, this runs EPANET 2.2.0
pressure = results.node['pressure']
flow = results.link['flowrate']
intial_pressures = pressure.iloc[0] #pressures at 0
intial_flow = abs(flow.iloc[0]) * 15850.3 # flow at 0, convert to gpm
ax_p = wntr.graphics.plot_network(wn, node_attribute=intial_pressures,title='Pre-Disaster Pressures',node_size=80,add_colorbar=True,node_colorbar_label='Pressure')
ax_f = wntr.graphics.plot_network(wn, link_attribute=intial_flow,title='Pre-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow')
#%%
wn_pipe_dmg = modify_inp_with_pipe_damage(df_pipeline_dmg_250, shp_guid_to_info, wn, pipeline_df)
wn_full_damaged = modify_inp_with_facility_damage(df_wterfclty_dmg_250, facility_mapping_dict, wn_pipe_dmg)

import json
leaky_node_dict = {}
for node_name, node in wn_full_damaged.nodes():
    if node._leak == True:
        leaky_node_dict[node_name] = node._leak_area
    else:
        leaky_node_dict[node_name] = 0
with open(f'leaky_node_{return_val}yr.json', 'w') as json_file:
    json.dump(leaky_node_dict, json_file)

obj = wntr.epanet.InpFile()
obj.write(f'damaged_network_{return_val}yr.inp',wn_full_damaged)
#%%
wn_full_damaged.options.hydraulic.demand_model = 'PDD'
sim_dmg = wntr.sim.EpanetSimulator(wn_full_damaged)
results_dmg = sim_dmg.run_sim() # by default, this runs EPANET 2.2.0
pressure_dmg = results_dmg.node['pressure']
flow_dmg = results_dmg.link['flowrate']
intial_pressures_dmg = pressure_dmg.iloc[0] #pressures at 0
intial_flow_dmg = abs(flow_dmg.iloc[0]) * 15850.3 # flow at 0, convert to gpm
ax_p_dmg = wntr.graphics.plot_network(wn, node_attribute=intial_pressures_dmg,title='Post-Disaster Pressures',node_size=80,add_colorbar=True,node_colorbar_label='Pressure')
ax_f_dmg = wntr.graphics.plot_network(wn, link_attribute=intial_flow_dmg,title='Post-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow')
#%% Plotting Difference

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
plt.figure() # Start new figure
# Sample DataSeries for x and y
x_data = intial_pressures
y_data = intial_pressures_dmg

# Create a scatter plot
plt.scatter(x_data, y_data, label='Scatter Plot', color='b', marker='o')
plt.plot(x_data, x_data, label='y = x', color='r') # True y=x line

# Add labels and title
plt.xlabel('Pre-Damage Pressures')
plt.ylabel('Post-Damage Pressures')
plt.title(f'Pressure Comparison Post-Disaster')

# Calculate R^2 value
r2 = r2_score(y_data, x_data)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_data, x_data))
# Display
plt.text(0.1, 0.9, f'R^2 = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.1, 0.85, f'RMSE = {rmse:.2f}', transform=plt.gca().transAxes, fontsize=12)

# Show the plot
plt.show()

### Plotting Flows
def nse(real_flows,simulated_flows):
   numerator = sum((obs - sim) ** 2 for obs, sim in zip(real_flows, simulated_flows))
   mean_observed = sum(real_flows) / len(real_flows)
   denominator = sum((obs - mean_observed) ** 2 for obs in real_flows)
   nse_value = 1 - (numerator / denominator)
   return nse_value

plt.figure() # Start new figure
# Sample DataSeries for x and y
x_data = intial_flow
y_data = intial_flow_dmg

# Create a scatter plot
plt.scatter(x_data, y_data, label='Scatter Plot', color='b', marker='o')
plt.plot(x_data, x_data, label='y = x', color='r') # True y=x line

# Add labels and title
plt.xlabel('Pre-Damage Flows, GPM')
plt.ylabel('Post-Damage Flows, GPM')
plt.title(f'Flow Comparison Post-Disaster')

# Calculate R^2 value
r2 = r2_score(y_data, x_data)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_data, x_data))
# Calculate NSE
nse = nse(x_data,y_data) #'real flows' are original simulation, simulated flows are diameter prediction
# Display
plt.text(0.1, 0.9, f'R^2 = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.1, 0.85, f'RMSE = {rmse:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.1, 0.80, f'NSE = {nse:.2f}', transform=plt.gca().transAxes, fontsize=12)

# Show the plot
plt.show()
