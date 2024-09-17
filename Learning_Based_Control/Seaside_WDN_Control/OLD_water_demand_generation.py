import wntr
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog
import networkx as nx
import numpy as np
import pandas as pd
import math
import json
import os
import matplotlib.pyplot as plt
import contextily as ctx
import random
from utils import water_demand_generation
#%%
def get_user_input():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.withdraw()
    return file_path

def reverse_dict(original_dict):
    reversed_dict = {}
    for key, value in original_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = [key]
        else:
            reversed_dict[value].append(key)
    return reversed_dict
#%% Shapefile loading
folder_path = r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\My Documents\GitHub\Seaside_WDN_Control\data"
plot = False
# print('Select EPANET file')
# epa_inp_file_path = get_user_input()
# wnet = wntr.network.WaterNetworkModel(epa_inp_file_path)

# Load in shapefiles
pipe_shpfile = f'{folder_path}/Seaside_water_pipelines_wgs84.shp'
node_shpfile = f'{folder_path}/Seaside_wter_nodes.shp'
bldg_geojson = f'{folder_path}/seaside_bldg.geojson'
census_blocks = gpd.read_file(f'{folder_path}/seaside_census_blocks.shp')
water_pipelines = gpd.read_file(pipe_shpfile)
water_node = gpd.read_file(node_shpfile)

buildings = gpd.read_file(bldg_geojson)
crs = 'EPSG:32610' # UTM Zone 10N for Oregon, code EPSG:32610 in meters
list_dem_paths = [f'{folder_path}/seaside_elevation_raster.tif']

# nodes, base_demand = water_demand_generation(water_pipelines, buildings,census_blocks,
#                                              list_dem_paths,crs,tourist_pop=2000,
#                                              building_snap_tolerance=700,plot=False)

#%% Read in file that has building populations, and assign tourists to seasonal rentals
# Read in data that has population assigned to buildings
bldg_pop_path = f'{folder_path}/seaside_bldg_pop.geojson'
bldg_pop = gpd.read_file(bldg_pop_path)

# Cleaning data from building geojson file
keep_columns = ['struct_typ','year_built','no_stories','appr_bldg','dgn_lvl','guid','origin',
                'rmv_improv','rmv_land','elev','strctid','numprec','ownershp','race','hispan',
                'vacancy','gqtype','livetype','landuse','geometry']
bldg_pop = bldg_pop[keep_columns]
fix_units = ['numprec','ownershp','race','hispan','vacancy','gqtype']
for col in fix_units:
    bldg_pop[col] = bldg_pop[col].replace('',np.nan)
    bldg_pop[col] = bldg_pop[col].astype(float) #6645 pop, now 7234.

commercial_occupied = False # assume night time, everyone at home
commercial_resident_pop_percent = 0.4 # Percentage of people not at home and at commercial buildings
tourist_population = 2000 # 6645 residents as assigned currently, commercial not occupied
# commercial_bldg = bldg_pop[bldg_pop['landuse'] == 'commercial']

if commercial_occupied:
    # Assign some residential population to commercial buildings
    print('Not implemented')
    # Assign some tourist population to commercial buildings
    
    # Assign tourist population to rentals
    
else:
    # Assign tourist population to rentals
    low_rental_min, low_rental_max = 1,5
    high_rental_min, high_rental_max = 10,40
    tourist_population = 2000
    t_list = []
    rental_bldgs = bldg_pop[bldg_pop['vacancy'] == 5].copy()
    shuffled_rentals = rental_bldgs.sample(frac=1).reset_index(drop=True).copy()
    bldg_guid_pop = {}
    for index, row in shuffled_rentals.iterrows():
        print(row['landuse'])
        if row['landuse'] == 'losr':
            if bldg_pop.loc[index, 'numprec'] == 0:
                people = random.randint(low_rental_min, low_rental_max)
                bldg_guid_pop[row['guid']] = people
                tourist_population -= people
                t_list.append(tourist_population)
        elif row['landuse'] == 'hosr':
            if bldg_pop.loc[index, 'numprec'] == 0:
                people = random.randint(high_rental_min, high_rental_max)
                bldg_guid_pop[row['guid']] = people
                tourist_population -= people
                t_list.append(tourist_population)
        else:
            assert row['landuse'] in ['losr','hosr'], 'Landuse must be losr or hosr for seasonal rentals'
        if tourist_population <= 0:
            print('Finished assigning tourist population to seasonal rentals')
            break
    if tourist_population > 0:
        print('Not enough tourists added on initial pass, looping again')
        while tourist_population > 0:
            for index, row in shuffled_rentals.iterrows():
                # Add more people to the high occupancy rentals
                if row['landuse'] == 'hosr':
                    existing_people = bldg_pop.loc[index, 'numprec']
                    people = random.randint(low_rental_min, low_rental_max)
                    bldg_guid_pop[row['guid']] = people  + existing_people
                    tourist_population -= people
                    t_list.append(tourist_population)
                    # Check if done adding tourist population
                if tourist_population <= 0:
                    print('Finished assigning tourist population to seasonal rentals')
                    break
    print(sum(list(bldg_guid_pop.values())))
bldg_pop.set_index('guid', inplace=True)
for index_value, population in bldg_guid_pop.items():
    if index_value in bldg_pop.index:
        bldg_pop.at[index_value, 'numprec'] = population
    else:
        print('error')

#%% might not need this! Assigning buildings to water nodes from GIS file
tolerance = 100
snap_buildings = wntr.gis.snap(bldg_pop, water_pipelines, tolerance)
snap_buildings['node'] = None
# Get all buildings nodes that are closer to the from node and assign the node index to them
temp = snap_buildings['line_position'] < 0.5
snap_buildings.loc[temp, 'node'] = water_pipelines.loc[snap_buildings.loc[temp, 'link'], 'fromnode'].values
# Get all buildings nodes that are closer to the to node and assign the node index to them
temp = snap_buildings['line_position'] >= 0.5
snap_buildings.loc[temp, 'node'] = water_pipelines.loc[snap_buildings.loc[temp, 'link'], 'tonode'].values
# Set population of water nodes to 0 and sum up population for all buildings assigned to that node
water_node.set_index('node_id',inplace=True)
water_node['pop'] = 0
for node_id, group_df in snap_buildings.groupby('node'):
    print(f'Node ID: {node_id}')
    print(group_df.head())
    total_pop = bldg_pop.loc[group_df.index,'numprec'].sum() 
    water_node.loc[node_id,'pop'] = total_pop
print('Lost population due to nan: ', sum(list(water_node[water_node['guid'].isna()]['pop'])))
water_node.dropna(subset=['guid'], inplace=True)
water_node['demand_gpm'] = water_node['pop'] * 80 / (24*60)  

#%% import epanet model
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
path = r"C:\Users\poffja\Box\Jason_Poff\INCORE_WNTR_Project\Data\Seaside_Data\FreshWaterNetwork\2021-07_Water_Network\bldgs2wter_Seaside.csv"
building2water = pd.read_csv(path)
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
path = r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\My Documents\GitHub\Seaside_WDN_Control\data\Seaside_dummy_model_fixed.inp"
wn = wntr.network.WaterNetworkModel(path) #load water network
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

#%% Assign new base demand from bldg_pop
demand_change_dict = {} # just for tracking
# give demand to junction 760?
demand_node_bldg_list_dict = reverse_dict(bldg_guid_demand_node_dict)
node_population_dictionary = {}
for node_name in wn.nodes:
    node_obj = wn.get_node(node_name)
    if node_obj.node_type == 'Junction':
        if node_name not in demand_node_bldg_list_dict:
            demand_node_bldg_list_dict[node_name] = [] # Assign empty list
        bldg_guid_list = demand_node_bldg_list_dict[node_name]
        # Get all populations assigned to these buildings and add them up
        total_pop = 0
        for bldg_guid in bldg_guid_list:
            total_pop += bldg_pop.loc[bldg_guid,'numprec']
        new_demand_gpm = total_pop * 80 / (24*60) # GPM 
        new_demand = new_demand_gpm  / 15850.323141488905 # to m^3/ sec
        demand_change_dict[node_name] = [node_obj.base_demand, new_demand]
        node_obj.demand_timeseries_list[0].base_value = new_demand # modify base demand value to new value.
#%% Simulation
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim() # by default, this runs EPANET 2.2.0
pressure = results.node['pressure']
flow = results.link['flowrate']
intial_pressures = pressure.iloc[0] #pressures at 0
intial_flow = abs(flow.iloc[0]) * 15850.3 # flow at 0, convert to gpm
ax_p = wntr.graphics.plot_network(wn, node_attribute=intial_pressures,title='Pre-Disaster Pressures',node_size=80,add_colorbar=True,node_colorbar_label='Pressure')
ax_f = wntr.graphics.plot_network(wn, link_attribute=intial_flow,title='Pre-Disaster Flows',link_width=2,node_size=0,add_colorbar=True,link_colorbar_label='Flow')

obj = wntr.epanet.InpFile()
obj.write(f'new_demands_seaside.inp',wn)