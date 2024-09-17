# Import modules
import numpy as np
import random
import math
import pandas as pd
import geopandas as gpd # For reading in shapefiles : another way. 
import os # For managing directories and file paths if drive is mounted
import matplotlib.pyplot as plt  # for plot.
import wntr
import generate_epa_model as gepa

from pyincore import IncoreClient, Dataset, DataService, HazardService, FragilityService
from pyincore import FragilityCurveSet, MappingSet
# from pyincore_viz.geoutil import GeoUtil as geoviz
# from pyincore_viz.plotutil import PlotUtil as plotviz
from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.pipelinedamage import PipelineDamage
from pyincore.analyses.waterfacilitydamage import WaterFacilityDamage
from pyincore.analyses.pipelinedamagerepairrate import PipelineDamageRepairRate
from pyincore.analyses.housingunitallocation import HousingUnitAllocation
from pyincore.analyses.cumulativebuildingdamage import CumulativeBuildingDamage
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
        random_float = np.random.random()
        facility_epa_name = facility_mapping_dict[row['guid']] # Assuming dict maps guid with epa pump/tank/reservoir name as values
        if row['haz_expose'] == 'yes':
            if random_float > row['LS_2']: # if damage exceeds LS0 and LS1, turn pump off.
                pump = wn.get_link(facility_epa_name)
                pump.add_outage(wn, break_time, fix_time)
                print(f'Modified Pump: {facility_epa_name}')
                pump_damage_info['damaged_pumps'].append(facility_epa_name)
        #This only works for pumps, assuming reservoirs and tanks have all damage probabilities set to 0. Seaside is ok cause no tanks
    return pump_damage_info

def default_water_network_sim(wn,full_graphs=False):
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
    leak_values = results_damaged.node['leak_demand'].iloc[1] * 15850.3
    damaged_pressure = results_damaged.node['pressure'].iloc[1] * 1.4219702063247 #pressures at 0
    damaged_flow = abs(results_damaged.link['flowrate'].iloc[1] * 15850.3) # flow at 0, convert to gpm
    damaged_velocity = abs(results_damaged.link['velocity'].iloc[1] * 3.28084) # velocity at 0, convert to ft/s
    damage_status = results_damaged.link['status'].iloc[1] 
    ### Graphing
    wntr.graphics.plot_network(wn, node_attribute=damaged_pressure,node_range=[0,120],title='Pressures',node_size=80,add_colorbar=True,node_colorbar_label='Pressure (PSI)')
    wntr.graphics.plot_network(wn, link_attribute=damage_status,title='Pipe Status',link_width=3,node_size=0,link_cmap=['red','blue'],add_colorbar=True,link_colorbar_label='Pipe Closed')
    if full_graphs:
        wntr.graphics.plot_network(wn, link_attribute=damaged_flow,title='Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow (GPM)')
        wntr.graphics.plot_network(wn, link_attribute=damaged_velocity,title='Velocities',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Velocity (ft/s)')
        wntr.graphics.plot_network(wn, node_attribute=leak_values,node_range=[0,None],title='Nodal Leaks',node_size=80,add_colorbar=True,node_colorbar_label='Leak Demand (GPM)')
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
    return results_damaged, fulfilled_avg_demand_ratio

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
#%% INCORE Damage simulation
load_results = True
folder_path = r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\My Documents\GitHub\Seaside_WDN_Control"
return_val = 1000

if not load_results:
    
    client = IncoreClient() # To get account, go to https://incore.ncsa.illinois.edu/doc/incore/account.html
    data_service = DataService(client)
    mapping_dict = {
    'buildings': 
        {'eq': '5d2789dbb9219c3c553c7977',
         'tsu': '5d279bb9b9219c3c553c7fba'},

    'water_pipe': 
        {'eq': '5b47c227337d4a38464efea8',
         'tsu': '5d320a87b9219c6d66398b45'},
    
    'water_facility': 
        {'eq': '6079f7825b609c56229bf976',
         'tsu': '6079b8a66799d908861e4bf0'},  # doesn't plot
}

    # Define hazard models: earthquake and tsunami
    #rt = [100, 250, 500, 1000, 2500, 5000, 10000]
    rt = [1000]
    
    eq_hazard_dict = {100: "5dfa4058b9219c934b64d495", 
                      250: "5dfa41aab9219c934b64d4b2",
                      500: "5dfa4300b9219c934b64d4d0",
                      1000: "5dfa3e36b9219c934b64c231",
                      2500: "5dfa4417b9219c934b64d4d3", 
                      5000: "5dfbca0cb9219c101fd8a58d",
                     10000: "5dfa51bfb9219c934b68e6c2"}
    
    tsu_hazard_dict = {100: "5bc9e25ef7b08533c7e610dc", 
                      250: "5df910abb9219cd00cf5f0a5",
                      500: "5df90e07b9219cd00ce971e7",
                      1000: "5df90137b9219cd00cb774ec",
                      2500: "5df90761b9219cd00ccff258",
                      5000: "5df90871b9219cd00ccff273",
                      10000: "5d27b986b9219c3c55ad37d0"}
    for ret_prd in rt: # loop through all return periods
        # for storage
        path_to_output = os.path.join(os.getcwd(), 'output', '{}yr' .format(ret_prd))
        
        ### Buildings_________________________________________
        # --- Earthquake
        # initializing building damage and fragility service
        bldg_dmg = BuildingDamage(client)   
        fragility_service = FragilityService(client)
        # defining building dataset (GIS point layer)
        bldg_dataset_id = "613ba5ef5d3b1d6461e8c415"
        bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)
        # specifiying mapping id from fragilites to building types
        mapping_id = "5d2789dbb9219c3c553c7977" # 4 DS
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        bldg_dmg.set_input_dataset('dfr3_mapping_set', mapping_set)
        bldg_dmg.set_parameter("hazard_type", "earthquake")
        bldg_dmg.set_parameter("num_cpu", 4)
        result_name = os.path.join(path_to_output, 'buildings_eq_{}yr' .format(ret_prd))
        hazard_id = eq_hazard_dict[ret_prd]
        bldg_dmg.set_parameter("hazard_id", hazard_id)
        bldg_dmg.set_parameter("result_name", result_name)
        bldg_dmg.run_analysis()
        print('Earthquake done.')
        
        # --- Tsunami
        # initializing pyincore building damage and fragility service 
        bldg_dmg = BuildingDamage(client)
        fragility_service = FragilityService(client)
        # defining building dataset (GIS point layer)
        bldg_dataset_id = "613ba5ef5d3b1d6461e8c415"
        bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)
        # specifiying mapping id from fragilites to building types
        mapping_id = "5d279bb9b9219c3c553c7fba" # 4 DS
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        bldg_dmg.set_input_dataset('dfr3_mapping_set', mapping_set)
        bldg_dmg.set_parameter("hazard_type", "tsunami")
        bldg_dmg.set_parameter("num_cpu", 4)
        result_name = os.path.join(path_to_output, 'buildings_tsu_{}yr' .format(ret_prd))
        hazard_id = tsu_hazard_dict[ret_prd]
        bldg_dmg.set_parameter("hazard_id", hazard_id)
        bldg_dmg.set_parameter("result_name", result_name)
        bldg_dmg.run_analysis()
        print('Tsunami done.')
        
        # --- Cumulative
        # initializing pyIncore cumulative building damage
        cumulative_bldg_dmg = CumulativeBuildingDamage(client)
        cumulative_bldg_dmg.set_parameter("num_cpu", 4)
        # reading in damage results from above analysis
        eq_damage_results_csv = os.path.join(path_to_output, 'buildings_eq_{}yr.csv' .format(ret_prd))
        tsu_damage_results_csv = os.path.join(path_to_output, 'buildings_tsu_{}yr.csv' .format(ret_prd))
        # loading datasets from CSV files into pyincore
        eq_damage_dataset = Dataset.from_file(eq_damage_results_csv, "ergo:buildingDamageVer5")
        tsu_damage_dataset = Dataset.from_file(tsu_damage_results_csv, "ergo:buildingDamageVer5")
        cumulative_bldg_dmg.set_input_dataset("eq_bldg_dmg", eq_damage_dataset)
        cumulative_bldg_dmg.set_input_dataset("tsunami_bldg_dmg", tsu_damage_dataset)
        # defining path to output 
        result_name = os.path.join(path_to_output, 'buildings_cumulative_{}yr' .format(ret_prd))
        cumulative_bldg_dmg.set_parameter("result_name", result_name)
        # running analysis
        cumulative_bldg_dmg.run_analysis()
        print('Cumulative done.')
    
        ### Water Pipelines____________________________ Earthquake only
        pipe_dataset_id = "60e72f9fd3c92a78c89636c7"
        wter_fclty_dataset_id = "60e5e91960b3f41243faa3b2"        # defining water facilities (GIS point layer)
        # Earthquake
        mapping_id = "5b47c227337d4a38464efea8"                  # specifiying mapping id from fragilites to pipe types
        fragility_service = FragilityService(client)             # loading fragility mapping
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        pipeline_dmg = PipelineDamageRepairRate(client)                    # initializing pyincore
        '''
        Pipeline Damage: calculated in number of repairs needed per km of pipes.
        2 types of damage:
        LEAKS: pull out at joints or crushed at bell, caused primarily by seismic wave propagation (PGV).
        BREAKS: pipe actually breaks, usually due to ground failure (PGD)
        Hazus Methodology assumes:
        seismic waves (PGV) 80% leaks and 20% breaks
        ground failure (PGD) 20% leaks and 80% breaks.
        SEASIDE: liquefaction probability very low, not considered. ONLY PGV!   
        '''
        pipeline_dmg.load_remote_input_dataset("pipeline", pipe_dataset_id) # loading in the above id
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        pipeline_dmg.set_input_dataset("dfr3_mapping_set", mapping_set)
        result_name = os.path.join(path_to_output, 'pipeline_eq_{}yr' .format(ret_prd))
        pipeline_dmg.set_parameter("hazard_type", "earthquake")  # defining hazard type (e.g. earthquake vs. tsunami)
        hazard_id = eq_hazard_dict[ret_prd]                   # specifying hazard id for specific recurrence interval
        pipeline_dmg.set_parameter("hazard_id", hazard_id)       # loading above into pyincore
        pipeline_dmg.set_parameter("fragility_key", "pgv")
        pipeline_dmg.set_parameter("num_cpu", 6)                 # number of CPUs to use for parallel processing
        pipeline_dmg.set_parameter("result_name", result_name)   # specifying output name in pyincore
        pipeline_dmg.run_analysis()                              # running the analysis with the above parameters: it gives you 
        '''
        Pipeline damage column meanings:
        pgvrepairs: num of repairs predicted /km due to seismic waves (PGV)
        pdgrepairs: num of repairs predicted /km due to ground displacement (PGD). 0- for Seaside
        repairspkm: Sum above ^ total number of repairs predicted /km
        breakrate: 0.2 * pgvrepairs + 0.8 *pdgrepairs
        leakrate: 0.8 * pgvrepairs + 0.2 *pdgrepairs
        failprob: 1 - math.exp(-1 * break_rate * length)
        numpgvrpr: pgvrepairs * length (km), how many repairs in that pipe due to pgv
        numpgdrpr: pgdrepairs * length (km), how many repairs in that pipe due to pgd
        numrepairs: ^ sum the number of repairs for pgv and pgd for each pipe
        '''
        
        ### Facility damage (pumps, tanks, reservoirs)
        # Earthquake
        fclty_dataset_id = "60e5e91960b3f41243faa3b2"             # defining Facility dataset (GIS layer);
        mapping_id = "5d39e010b9219cc18bd0b0b6"                  # specifiying mapping id from fragilites to pipe types
        fragility_service = FragilityService(client)             # loading fragility mapping
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        
        wterfclty_dmg = WaterFacilityDamage(client)                    # initializing pyincore
        wterfclty_dmg.load_remote_input_dataset("water_facilities", fclty_dataset_id) # loading in the above id
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        wterfclty_dmg.set_input_dataset("dfr3_mapping_set", mapping_set)
        result_name = os.path.join(path_to_output, 'wterfclty_eq_{}yr' .format(ret_prd))
        wterfclty_dmg.set_parameter("hazard_type", "earthquake")  # defining hazard type (e.g. earthquake vs. tsunami)
        hazard_id = eq_hazard_dict[ret_prd]                   # specifying hazard id for specific recurrence interval
        wterfclty_dmg.set_parameter("hazard_id", hazard_id)       # loading above into pyincore
        wterfclty_dmg.set_parameter("fragility_key", "pga")
        wterfclty_dmg.set_parameter("num_cpu", 6)                 # number of CPUs to use for parallel processing
        wterfclty_dmg.set_parameter("result_name", result_name)   # specifying output name in pyincore
        wterfclty_dmg.run_analysis()                              # running the analysis with the above parameters: it gives you 
        
    
else:
    df_pipeline_dmg = pd.read_csv(f'{folder_path}/damage_data/damage_dfs/pipeline_eq_{return_val}yr_pipeline_result.csv')
    df_wterfclty_dmg = pd.read_csv(f'{folder_path}/damage_data/damage_dfs/wterfclty_eq_{return_val}yr_wterfclty_result.csv')

#%% Create water network

# Load shapefiles
crs = 'EPSG:32610' # UTM Zone 10N for Oregon, code EPSG:32610 in meters
bldg_pop_path = f'{folder_path}/data/seaside_bldg_pop.geojson'
bldg_pop = gpd.read_file(bldg_pop_path)
# Add tourists to seaside
bldg_pop_adj = gepa.seaside_bldg_pop_adjustment(bldg_pop, crs, tourist_population = 2000)

# Load shapefiles
pipe_shpfile = f'{folder_path}/data/Seaside_water_pipelines_wgs84.shp'
node_shpfile = f'{folder_path}/data/Seaside_wter_nodes.shp'
list_dem_paths = [f'{folder_path}/data/seaside_elevation_raster.tif']
water_pipelines = gpd.read_file(pipe_shpfile)
water_nodes = gpd.read_file(node_shpfile)
#  Create water network with adjusted bldg pop
wn = gepa.create_seaside_wn(bldg_pop_adj, water_pipelines, water_nodes, list_dem_paths, crs)

# wn = wntr.network.WaterNetworkModel(r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\My Documents\GitHub\Seaside_WDN_Control\my_seaside.inp")
    
### Leak Testing: This proves that leaks work, just need to reset the model from scratch it seems.
# wn = wntr.network.WaterNetworkModel(r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\My Documents\GitHub\Seaside_WDN_Control\my_seaside.inp")
# wn.options.time.duration = 2*3600 
# wn.options.hydraulic.headloss = 'H-W'
# wn.options.hydraulic.demand_model = 'PDD'
# wn.options.hydraulic.minimum_pressure = 3.516 # 5 PSI.  1 PSI to 0.7032496149020262 M
# wn.options.hydraulic.required_pressure = 24.614 # 35 PSI
# wn.options.hydraulic.pressure_exponent = 0.5
# j320 = wn.get_node('J-320')
# tot_pipe_area = np.pi * .2032 ** 2 / 4
# j320.remove_leak(wn)
# j320.add_leak(wn, area=tot_pipe_area*0.2, start_time=0, end_time=None)
# sim = wntr.sim.WNTRSimulator(wn)
# results = sim.run_sim() # by default, this runs EPANET 2.2.0
# j320_d = results.node['demand']['J-320']
# j320_ld = results.node['leak_demand']['J-320']
# p50_f = results.link['flowrate']['P-50']

#%% Create mappings for damage translation

if water_pipelines.index.name != 'guid':
    water_pipelines.set_index('guid',inplace=True)
pipe_mapping_dict = { guid_index: f'P-{row["Link_ID"]}' for guid_index, row in water_pipelines.iterrows()} 
# Mapping for facilities  
facility_mapping_dict = {'2048ed28-395e-4521-8fc5-44322534592e' : 'R-1',
 '8d22fef3-71b6-4618-a565-955f4efe00bf' : 'TillaHPHSE',
 'cfe182a2-c39c-4734-bcd5-3d7cadab8aff' : 'PHouse',
 'd6ab5a29-1ca1-4096-a3c3-eb93b2178dfe' : 'RegalHPHSE'}

#%% Translate pipe damage
# Pipe Damage
pipe_node_damage_info = modify_inp_with_pipe_damage(df_pipeline_dmg, 
                                                                 pipe_mapping_dict, 
                                                                 wn, break_percent=0.4,
                                                                 leak_percent=0.1,
                                                                 break_time=1*3600,
                                                                 fix_time=2*3600,
                                                                 close_pipes=True)
pump_damage_info = modify_inp_with_facility_damage(df_wterfclty_dmg, facility_mapping_dict, wn)

### Various Checks
# Leaks check: get dictionary with nodes and leaks assigned
# leaky_node_dict = {}
# count = 0
# for node_name, node in wn.nodes():
#     if node._leak == True:
#         leaky_node_dict[node_name] = node._leak_area
#         count +=1
#     else:
#         leaky_node_dict[node_name] = 0
# print(f'{count} nodes with leaks added')  

#%% See how networks performs if no one moves?

pre_move_results, ratio_avg_water_demand_met = default_water_network_sim(wn)
print(f'Percent of Average Demand Met: {round(ratio_avg_water_demand_met*100,1)}%')

#%% Calculate population shift due to building damage or tsunami evacuation and get new demand values
plot = False
pop_shift = 'tsunami_evac' # or 'post-tsunami'

if pop_shift == 'tsunami_evac':
    ### OPTION 1: TSUNAMI EVACUATION. All residents/tourists flee to designated areas
    
    # Load evacuation zone polygons and building shpfile
    crs = 'EPSG:32610' # UTM Zone 10N for Oregon, code EPSG:32610 in meters
    evac_zone_polygons = gpd.read_file(f'{folder_path}/data/Evacuation_Zones.shp')
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
            
    # Get new water network
    water_pipelines = gpd.read_file(pipe_shpfile)
    tsunami_evac_wn = gepa.create_seaside_wn(buildings_within_evac_zones, water_pipelines, water_nodes, list_dem_paths, crs)
    # obj = wntr.epanet.InpFile()
    # obj.write(f'tsunami_evac_seaside.inp',tsunami_evac_wn)
    
    # Apply the same damage to the water network
    apply_damage_to_wn(tsunami_evac_wn, pipe_node_damage_info, pump_damage_info,break_time=1*3600,fix_time=2*3600)
    # Power up all pumps, some might be turned off. small pumps typically 250 - 1100W, bigger can be 2000-5000W
    tsunami_evac_wn.get_link('RegalHPHSE').power = 1000 #regal 0.005 HP, 3.677 W
    tsunami_evac_wn.get_link('PHouse').power = 1000 #phouse 0.3 HP, 220.65 W
    tsunami_evac_wn.get_link('TillaHPHSE').power = 1000# tillamook 0.15 HP, 110.325 W
    # Do simulation analysis
    tsunami_results, tsu_ratio_avg_water_demand_met = default_water_network_sim(tsunami_evac_wn)
    print(f'Percent of Average Demand Met: {round(tsu_ratio_avg_water_demand_met*100,1)}%')


if pop_shift == 'post_tsunami':
    ### OPTION 2: Post tsunami, some residents can return home (if home not damaged), 
    ###           tourists leave city, others stay in shelters (schools?)
    
    # Load building shpfile and building damage
    crs = 'EPSG:32610' # UTM Zone 10N for Oregon, code EPSG:32610 in meters
    bldg_pop_adj.to_crs(crs,inplace=True)
    
        
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
            
    # Get new water network
    water_pipelines = gpd.read_file(pipe_shpfile)
    tsunami_evac_wn = gepa.create_seaside_wn(buildings_within_evac_zones, water_pipelines, water_nodes, list_dem_paths, crs)
    # obj = wntr.epanet.InpFile()
    # obj.write(f'tsunami_evac_seaside.inp',tsunami_evac_wn)
    
    # Apply the same damage to the water network
    apply_damage_to_wn(tsunami_evac_wn, pipe_node_damage_info, pump_damage_info,break_time=1*3600,fix_time=2*3600)
    # Power up all pumps, some might be turned off. small pumps typically 250 - 1100W, bigger can be 2000-5000W
    tsunami_evac_wn.get_link('RegalHPHSE').power = 1000 #regal 0.005 HP, 3.677 W
    tsunami_evac_wn.get_link('PHouse').power = 1000 #phouse 0.3 HP, 220.65 W
    tsunami_evac_wn.get_link('TillaHPHSE').power = 1000# tillamook 0.15 HP, 110.325 W
    # Do simulation analysis
    tsunami_results, tsu_ratio_avg_water_demand_met = default_water_network_sim(tsunami_evac_wn)
    print(f'Percent of Average Demand Met: {round(tsu_ratio_avg_water_demand_met*100,1)}%')



