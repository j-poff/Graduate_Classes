# -*- coding: utf-8 -*-

import fiona, os, rasterio, momepy, wntr, scipy
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from functools import partial

from pyproj import CRS, Transformer
from shapely.ops import transform


from pyincore import IncoreClient, Dataset, DataService, HazardService, FragilityService
from pyincore import FragilityCurveSet, MappingSet
#from pyincore_viz.geoutil import GeoUtil as geoviz
#from pyincore_viz.plotutil import PlotUtil as plotviz

from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.pipelinedamage import PipelineDamage
from pyincore.analyses.waterfacilitydamage import WaterFacilityDamage
from pyincore.analyses.pipelinedamagerepairrate import PipelineDamageRepairRate

from pyincore.analyses.housingunitallocation import HousingUnitAllocation


#%%
def Explode_LineStrings(mains):
    mains_explode = dict()
    
    pipe_counter = 1
    for pid in mains.index:
        # if pipe_counter == 3775:
        #     stp = 0
        
        if list(mains.loc[[pid]].geometry.type)[0] == 'LineString':
            lines_all = list(mains.loc[[pid]].geometry.iloc[0].coords)

            for i in range(0,len(lines_all)-1):

                mains_explode[pipe_counter]= dict()
                mains_explode[pipe_counter]['Index'] = pipe_counter
                for col in mains.columns:
                    if col == 'geometry':
                        mains_explode[pipe_counter][col] = LineString([lines_all[i][0:2], lines_all[i+1][0:2]])
                    else:
                        mains_explode[pipe_counter][col] = list(mains.loc[[pid],col])[0]
                pipe_counter += 1
                
                
        else:
            for singline in mains.loc[[pid]].geometry.explode(index_parts=True):
                for i in range(0,len(singline.coords)-1):
                    
                    mains_explode[pipe_counter]= dict()
                    mains_explode[pipe_counter]['Index'] = pipe_counter
                    for col in mains.columns:
                        if col == 'geometry':
                            mains_explode[pipe_counter][col] = LineString([singline.coords[i][0:2], singline.coords[i+1][0:2]])
                        else:
                            mains_explode[pipe_counter][col] = list(mains.loc[[pid],col])[0]
                    pipe_counter += 1
                    
   
    df = pd.DataFrame.from_dict(mains_explode, orient='index')
  
    return gpd.GeoDataFrame(df, geometry=df.geometry, crs = mains.crs)

#%%
def Explode_MultiLineStrings(mains):
    mains_explode = dict()
    
    pipe_counter = 1
    for pid in mains.index:
        
        if list(mains.loc[[pid]].geometry.type)[0] == 'LineString':

            mains_explode[pipe_counter]= dict()
            mains_explode[pipe_counter]['Index'] = pipe_counter
            for col in mains.columns:
                if col == 'geometry':
                    mains_explode[pipe_counter][col] = mains.loc[[pid]].geometry.iloc[0]
                else:
                    # print(list(mains.loc[[pid],col]))
                    mains_explode[pipe_counter][col] = list(mains.loc[[pid],col])[0]
            pipe_counter += 1
                
        else:
            for singline in mains.loc[[pid]].geometry.explode():
                
                mains_explode[pipe_counter]= dict()
                mains_explode[pipe_counter]['Index'] = pipe_counter
                for col in mains.columns:
                    if col == 'geometry':
                        mains_explode[pipe_counter][col] = singline
                    else:
                        mains_explode[pipe_counter][col] = list(mains.loc[[pid],col])[0]
            pipe_counter += 1
                    
   
    df = pd.DataFrame.from_dict(mains_explode, orient='index')
  
    return gpd.GeoDataFrame(df, geometry=df.geometry, crs = mains.crs)

#%%
def subgraph_to_geodataframes(G, crs=None):
    node_link_data = nx.node_link_data(G)

    nodes = gpd.GeoDataFrame(node_link_data['nodes'])
    nodes['geometry'] = [Point(xy) for xy in nodes.id]

    links = gpd.GeoDataFrame(node_link_data['links'])
    # assign start and end node names
    links['start_node_name'] = links['source']
    links['end_node_name'] = links['target']
    
    # I think this issue with invalid geometries has been resolved with the use of ox.get_undirected(G)
    invalid_link_geom = ~(links['geometry'].geom_type).isin(['LineString', 'MultiLineString'])
    if invalid_link_geom.sum() > 0:
        links['Valid Geometry'] = True
        print('Warning: Invalid link geometries', invalid_link_geom.sum())
        print('         Filled with straight line segments')
        for i, row in links.loc[invalid_link_geom,:].iterrows():
            #print(row['geometry'])
            node1_geom = nodes.loc[nodes['id']==row['from'],'geometry']
            node2_geom = nodes.loc[nodes['id']==row['to'],'geometry']
            links.loc[i, 'geometry'] = LineString([(node1_geom.x.values[0],
                                                    node1_geom.y.values[0]),
                                                   (node2_geom.x.values[0],
                                                    node2_geom.y.values[0])])
            links.loc[i, 'Valid Geometry'] = False
            
    links.set_crs(crs, inplace=True)
    nodes.set_crs(crs, inplace=True)

    return nodes, links    

#%%
# box_incore_path = r'C:\Users\riversam\Box\Research\INCORE_WNTR'
box_incore_path = r'C:\Users\poffja\Box\Jason_Poff\INCORE_WNTR'
client = IncoreClient()
data_service = DataService(client)


#%% Readign the building inventory for the city of Seaside. 
bldg_dataset_id = "613ba5ef5d3b1d6461e8c415"        # defining building dataset (GIS point layer)
bldg_dataset = Dataset.from_data_service(bldg_dataset_id, data_service)
#geoviz.plot_map(bldg_dataset, column='struct_typ',category='True')
bldg_df = bldg_dataset.get_dataframe_from_shapefile()
bldg_df.set_index('guid', inplace=True)
print('Number of buildings: {}' .format(len(bldg_df)))

#%% Get housing and population in the buildings
# Create housing allocation 
hua = HousingUnitAllocation(client)

# Load input dataset
housing_unit_inv_id = "5d543087b9219c0689b98234"
address_point_inv_id = "5d542fefb9219c0689b981fb"
bldg_inv_id = "613ba5ef5d3b1d6461e8c415" 

hua.load_remote_input_dataset("housing_unit_inventory", housing_unit_inv_id)
hua.load_remote_input_dataset("address_point_inventory", address_point_inv_id)
hua.load_remote_input_dataset("buildings", bldg_inv_id)  
  
# Set analysis parameters
path_out = os.path.join(os.getcwd(), 'hua_0')
hua.set_parameter("result_name", path_out)
hua.set_parameter("seed", 1337)
hua.set_parameter("iterations", 1)
hua.run_analysis() # changed Line 307 from dataset.py from the pyincore package to filename = u'\\\\?\\' + self.get_file_path('csv')

# Retrieve result dataset
hua_result = hua.get_output_dataset("result")
# Convert dataset to Pandas DataFrame
hua_df = hua_result.get_dataframe_from_csv(low_memory=False)
# keep observations where the housing unit characteristics have been allocated to a structure.
# hua_df = hua_df.dropna(subset=['guid'])

#%% Reading the water infrastrcuture: water pipes, a water treatment plant, and three water pumping stations. 
wter_pipe_dataset_id = "60e72f9fd3c92a78c89636c7"        # defining water pipes (GIS point layer)
wter_pipe_dataset = Dataset.from_data_service(wter_pipe_dataset_id, data_service)
#geoviz.plot_map(wter_pipe_dataset, column=None, category=False)

wter_fclty_dataset_id = "60e5e91960b3f41243faa3b2"        # defining water facilities (GIS point layer)
wter_fclty_dataset = Dataset.from_data_service(wter_fclty_dataset_id, data_service)
#geoviz.plot_map(wter_fclty_dataset, column='utilfcltyc', category=True)

bldg_wter_map = pd.read_csv(box_incore_path + '\Data\Seaside_Data\FreshWaterNetwork/2021-07_Water_Network/bldgs2wter_Seaside.csv').set_index('bldg_guid').to_dict(orient='index')

#%%
wter_pipe_gdf = gpd.read_file(wter_pipe_dataset.local_file_path)

temp_node_coords = []
for i, row in wter_pipe_gdf.iterrows():

    
    if row['geometry'].geom_type == 'MultiLineString':
        cc = 0
        for singleline in wter_pipe_gdf.loc[[i]].geometry.explode():
            if cc == 0:
                xx, yy = list(singleline.coords.xy)
                temp_node_coords.append((row['Link_ID'], row['fromnode'], xx[0], yy[0]))
                cc = 1

        xx, yy = list(singleline.coords.xy)
        temp_node_coords.append((row['Link_ID'], row['tonode'], xx[-1], yy[-1]))   
            
    else:
        xx, yy = list(row['geometry'].coords.xy)
    
        temp_node_coords.append((row['Link_ID'], row['fromnode'], xx[0], yy[0]))
        temp_node_coords.append((row['Link_ID'], row['tonode'], xx[-1], yy[-1]))

utemp_node_coords = np.unique(np.asarray(temp_node_coords),axis=0)
ind = np.argsort(utemp_node_coords[:,1])
utemp_node_coords = utemp_node_coords[ind]

new_node_coords = []
new_node_dict = dict()
for nodei in np.unique(utemp_node_coords[:,1]):
    
    temp_ind = list(np.where(utemp_node_coords[:,1] == nodei)[0])
    
    temp_x = [utemp_node_coords[jx][2] for jx in temp_ind]
    temp_y = [utemp_node_coords[jy][3] for jy in temp_ind]
    
    new_node_coords.append((nodei, np.mean(temp_x), np.mean(temp_y)))
    #new_node_coords.append((nodei, list(scipy.stats.mode(temp_x))[0][0], list(scipy.stats.mode(temp_y))[0][0]))
    new_node_dict[int(nodei)]=dict()
    new_node_dict[int(nodei)]['Index'] = nodei
    new_node_dict[int(nodei)]['geometry'] = Point([np.mean(temp_x), np.mean(temp_y), 0])
    #new_node_dict[int(nodei)]['geometry'] = Point([list(scipy.stats.mode(temp_x))[0][0], list(scipy.stats.mode(temp_y))[0][0], 0])
    
new_node_coords = np.asarray(new_node_coords)
#temp_dist_nodes = scipy.spatial.distance_matrix(new_node_coords, new_node_coords)
df = pd.DataFrame.from_dict(new_node_dict, orient='index')
df_nodes = gpd.GeoDataFrame(df, geometry = df.geometry, crs = wter_pipe_gdf.crs)  

#%%
wter_pipe_df = pd.DataFrame(wter_pipe_gdf)
wter_pipe_df = wter_pipe_df.drop(columns='geometry')
new_geometry_dict = dict()

for i, row in wter_pipe_gdf.iterrows():
    
    temp_ini_node = np.where(new_node_coords[:,0] == int(row['fromnode']))[0]
    temp_end_node = np.where(new_node_coords[:,0] == int(row['tonode']))[0]
             
    if row['geometry'].geom_type == 'MultiLineString':
        
        cc = 0
        temp_multiline_coords = []
        for singleline in wter_pipe_gdf.loc[[i]].geometry.explode():
        #for singleline in wter_pipe_df.loc[[i]]['geometry'].explode():
            temp_singleline_coords = []
            xx, yy = list(singleline.coords.xy)
            
            if cc == 0:
                for ii in range(0,len(xx)):
                    if ii == 0:
                        temp_singleline_coords.append([new_node_coords[temp_ini_node][0][1], new_node_coords[temp_ini_node][0][2]])
                    else:
                        temp_singleline_coords.append([xx[ii], yy[ii]])
                
            elif cc == len(wter_pipe_gdf.loc[[i]].geometry.explode()):
                for ii in range(0,len(xx)):
                    if ii == len(xx):
                        temp_singleline_coords.append([new_node_coords[temp_end_node][0][1], new_node_coords[temp_end_node][0][2]])
                    else:
                        temp_singleline_coords.append([xx[ii], yy[ii]])
                    
            else:
                for ii in range(0,len(xx)):
                    temp_singleline_coords.append([xx[ii], yy[ii]])
                        
            temp_multiline_coords.append(temp_singleline_coords)
            cc = cc + 1

        print(i)
        new_geometry_dict[i] = MultiLineString(temp_multiline_coords) 
        
    else:
        temp_ini_coords = (new_node_coords[temp_ini_node][0][1], new_node_coords[temp_ini_node][0][2])
        temp_end_coords = (new_node_coords[temp_end_node][0][1], new_node_coords[temp_end_node][0][2])
        #new_geometry_dict[i] = LineString([temp_ini_coords, temp_end_coords])
        lines_all = list(row['geometry'].coords)
        if len(lines_all) == 2:
            #wter_pipe_df.loc[[i]]['geometry'] = LineString([temp_ini_coords, temp_end_coords])     
            new_geometry_dict[i] = LineString([temp_ini_coords, temp_end_coords])
        else:
            temp_singleline_coords = []
            for ii in range(0,len(lines_all)):
                if ii == 0:
                    temp_singleline_coords.append(temp_ini_coords)
                elif ii == len(lines_all)-1:
                    temp_singleline_coords.append(temp_end_coords)
                else:
                    temp_singleline_coords.append((lines_all[ii][0],lines_all[ii][1]))
            
            new_geometry_dict[i] = LineString(temp_singleline_coords)

wter_pipe_df['geometry'] = new_geometry_dict
wter_pipe_gdf_new = gpd.GeoDataFrame(wter_pipe_df, geometry = wter_pipe_df.geometry, crs = wter_pipe_gdf.crs)
#wter_pipe_gdf = Explode_MultiLineStrings(wter_pipe_gdf)
wter_pipe_gdf_exp = Explode_LineStrings(wter_pipe_gdf_new)

wter_pipe_graph = momepy.gdf_to_nx(wter_pipe_gdf_exp, approach='primal')

#%%
# ax1 = wter_pipe_gdf_exp.plot()
# df_nodes.plot(ax=ax1)
# #df_nodes.loc[[344]].plot(ax=ax1, color='y')
# df_nodes.apply(lambda x: ax1.annotate(text=x['Index'], xy=x.geometry.centroid.coords[0], ha='center', color='red'), axis=1)

#%%
wds_nodes_df, wds_edges_df, sw = momepy.nx_to_gdf(wter_pipe_graph, points=True, lines=True, spatial_weights=True)

sub_graphs = list(wter_pipe_graph.subgraph(c) for c in nx.connected_components(wter_pipe_graph))
max_graph = [0,0]
for i in range(0,len(sub_graphs)):
    if max_graph[1] < len(sub_graphs[i]):
        max_graph = [i,len(sub_graphs[i])]

# Store the orginial max graph, and create one with all graphs used for modifying
max_sub_graph_org = nx.Graph(sub_graphs[max_graph[0]])
max_sub_graph_mod = nx.Graph(wter_pipe_graph)

sub_graph_bool = 0
iter_count = 0

# Loop through to connect each subgraph
while sub_graph_bool == 0:
#while iter_count <= 6:
    if iter_count > 0:
        sub_graphs = list(max_sub_graph_mod.subgraph(c) for c in nx.connected_components(max_sub_graph_mod))
    
    # If there is only one graph left, stop the loop. Everything is connected.
    if len(sub_graphs) == 1:
        sub_graph_bool = 1
    
    # More than one graph left, must continue connecting
    if len(sub_graphs) > 1:
        max_graph = [0,0]
        for i in range(0,len(sub_graphs)):
            if max_graph[1] < len(sub_graphs[i]):
                max_graph = [i,len(sub_graphs[i])]
                
        node_coords = []
        for ni in sub_graphs[max_graph[0]].nodes():
            node_coords.append(np.asarray(ni))
                
        outer_nodes_distance = dict()
        added_nodes = dict()
        
        min_sub_graph_dist = float('inf')
        for i in range(0,len(sub_graphs)):

            if i != max_graph[0]:
                outer_nodes_distance[i] = dict()
                
                sub_min_dist = float('inf')
                min_dist = float('inf')
                for n, d in sub_graphs[i].degree():
                    if d == 1 and n not in added_nodes.keys():
                        #temp_dist = nx.shortest_path_length(road_graph, target=n, weight='mm_len')
                        temp_dist = scipy.spatial.distance_matrix(np.asmatrix(n), np.asmatrix(node_coords), p=2)
                        min_dist = temp_dist[0,np.argsort(temp_dist)[0][0]]
        
                        if min_dist < sub_min_dist:
                            sub_min_dist = min_dist
                            outer_nodes_distance[i]['node'] = [n, tuple(node_coords[np.argsort(temp_dist)[0][0]])]
                                    
                if sub_min_dist < min_sub_graph_dist:
                    min_sub_graph_dist = sub_min_dist
                    
                outer_nodes_distance[i]['min_dist'] = sub_min_dist
                
        edge_attr = list(list(sub_graphs[max_graph[0]].edges(data=True))[0][-1].keys())

        for i in outer_nodes_distance.keys():
            if outer_nodes_distance[i]['min_dist'] == min_sub_graph_dist:
                temp_edge = sub_graphs[i].edges(outer_nodes_distance[i]['node'][0])
                temp_pipe_attr = sub_graphs[i].get_edge_data(list(temp_edge)[0][0],list(temp_edge)[0][1])                    
                temp_edge_attr = dict()
                for e_attr in edge_attr:
                                        
                    if e_attr != 'geometry' and e_attr != 'length_km':
                        if isinstance(temp_pipe_attr, dict):
                            temp_edge_attr[e_attr] = temp_pipe_attr[e_attr]
                        else:
                            temp_edge_attr[e_attr] = temp_pipe_attr._atlas[list(temp_pipe_attr._atlas.keys())[0]][e_attr]
                    
                    elif e_attr == 'geometry':
                        temp_edge_attr['geometry'] = LineString([outer_nodes_distance[i]['node'][0], outer_nodes_distance[i]['node'][1]])
                    
                    elif e_attr == 'length_km':
                        temp_edge_attr[e_attr] = -999.0
               
                attrs = {(outer_nodes_distance[i]['node'][0], outer_nodes_distance[i]['node'][1]): temp_edge_attr}
                max_sub_graph_mod.add_edge(outer_nodes_distance[i]['node'][0], outer_nodes_distance[i]['node'][1])
                nx.set_edge_attributes(max_sub_graph_mod, attrs)
                
    iter_count += 1

wter_pipe_graph = nx.Graph(max_sub_graph_mod)

#%%
total_pop = 0
pipe_demand = dict()
for i, row in hua_df.iterrows():
    if not np.isnan(row['numprec']) and row['guid'] in bldg_wter_map.keys():
        temp_pipe_id = bldg_wter_map[row['guid']]['edge_guid']
        if temp_pipe_id not in pipe_demand.keys():
            pipe_demand[temp_pipe_id] = dict()
            pipe_demand[temp_pipe_id]['abs_sum'] = 0
            pipe_demand[temp_pipe_id]['per_total'] = 0
        pipe_demand[temp_pipe_id]['abs_sum'] += float(row['numprec'])
        total_pop += float(row['numprec'])
        
# assumed 100 gpd/per
for i in pipe_demand.keys():
    #pipe_demand[i]['per_total'] = (pipe_demand[i]['abs_sum']/total_pop)*100
    pipe_demand[i]['per_total'] = pipe_demand[i]['abs_sum']*100
        
#%%
node_demand = dict()
for i in pipe_demand.keys():
    #temp_row = wds_edges_df.loc[wds_edges_df['guid']==str(i)] # Explode MultiLineStrings
    temp_row = wds_edges_df.loc[wds_edges_df['guid']==str(i)].iloc[0] # Explode LineStrings
    if int(temp_row['node_start']) not in node_demand.keys():
        node_demand[int(temp_row['node_start'])] = 0
    
    node_demand[int(temp_row['node_start'])] += pipe_demand[i]['per_total']/2
    
    if int(temp_row['node_end']) not in node_demand.keys():
            node_demand[int(temp_row['node_end'])] = 0
        
    node_demand[int(temp_row['node_end'])] += pipe_demand[i]['per_total']/2
        
node_attr_dict = dict()
for i, row in wds_nodes_df.iterrows():
    if row['nodeID'] in node_demand.keys():
        node_attr_dict[list(row['geometry'].coords)[0]] = {'base_demand': node_demand[row['nodeID']]}
    else:
        node_attr_dict[list(row['geometry'].coords)[0]] = {'base_demand': 0.0}

nx.set_node_attributes(wter_pipe_graph, node_attr_dict)

#%%
dem_path = box_incore_path + '/Data/Seaside_Data_added/USGS_DEM'
dem_name = 'Seaside_DEM_Proj.tif'    
src_dem = rasterio.open(dem_path + '/' + dem_name)

coords = []
for ni in wter_pipe_graph.nodes():
    coords.append(ni)

node_elev = [x[0] for x in src_dem.sample(coords)]

elev_dict = dict()
for i in range(0,len(coords)):
    elev_dict[coords[i]] = {'elevation': node_elev[i]}

nx.set_node_attributes(wter_pipe_graph, elev_dict)

#%%
## setting up data to be read into wntr
pipe_roughness = pd.read_csv(box_incore_path + '\Data\Seaside_Data\FreshWaterNetwork/2021-07_Water_Network/Pipe_material_to_roughness.csv').set_index('Pipe_material').to_dict(orient='index')

# initialize gisdata dictionary
wn_keys = ['junctions', 'tanks', 'reservoirs', 
    'pipes', 'pumps', 'valves']

gisdata = dict.fromkeys(wn_keys, gpd.GeoDataFrame())

# get node/edge data post-contracting
nodes, edges = momepy.nx_to_gdf(wter_pipe_graph, points=True, lines=True)
edges['Index'] = edges.index.values

pipe_properties = dict()
pipe_properties['diameter'] = dict()
pipe_properties['roughness'] = dict()
pipe_properties['length'] = dict()

project_to_meter = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32610), always_xy=True)

for rid, row in  edges.iterrows():
    if row['diameter'] > 0:
        pipe_properties['diameter'][rid] = np.round(row['diameter']/39.3701, 6)
    else:
        pipe_properties['diameter'][rid] = 12.0/39.3701
    
    if row['Pipe_type'] in pipe_roughness.keys():
        pipe_properties['roughness'][rid] = pipe_roughness[row['Pipe_type']]['Roughness (feet)']*1000
    else:
        pipe_properties['roughness'][rid] = 0.00015*1000
        
    line2 = transform(project_to_meter.transform, row['geometry'])
    pipe_properties['length'][rid] = line2.length*3.28084

    # if row['length_km'] < -1:
    #     line2 = transform(project_to_meter.transform, row['geometry'])
    #     pipe_properties['length'][rid] = line2.length*3.28084
    
    # else:
    #     pipe_properties['length'][rid] = row['length_km']*3280.84
    

edges['diameter'] = pipe_properties['diameter']
edges['roughness'] = pipe_properties['roughness']
edges['length'] = pipe_properties['length']

# relabel columns appropriately
edges.rename(columns = {'node_start' : 'start_node_name', 'node_end' : 'end_node_name'}, inplace = True)

# add node_type column to nodes
nodes.insert(loc = 0, column = 'node_type', value = 'Junction')

# add link_type column to edges
edges.insert(loc = 0, column = 'link_type', value = "Pipe")

#change node names to strings (wntr wants strings)
nodes.index = nodes.index.map(str)
edges['start_node_name'] = edges['start_node_name'].apply(str)
edges['end_node_name'] = edges['end_node_name'].apply(str)

# change pipe names to strings
edges.index = edges.index.map(str)

gisdata['pipes'] = edges
gisdata['junctions'] = nodes

#%%
pattern = [1] * 24

# create water network from gis data
wn = wntr.network.io.from_gis(gisdata)
wn.add_pattern('1', pattern)

wn.options.hydraulic.demand_model = 'DD'
wn.options.hydraulic.headloss = 'D-W'
wn.options.hydraulic.required_pressure = 20
wn.options.hydraulic.minimum_pressure = 0
wn.options.time.duration = 1*3600

wntr.graphics.plot_network(wn, node_attribute='elevation', node_colorbar_label='Elevation (m)')

for node_id in list(wn.junction_name_list):
    junction = wn.get_node(node_id)
    junction.demand_timeseries_list[0].base_value = float(nodes.iloc[int(junction.name)]['base_demand']*0.0006944)/15850.32314147059
    junction.demand_timeseries_list[0].pattern_name = '1'
    

#%% Hardcoded parameters

resv_name = 'R-1'
base_head = 215/3.280839895
coordinates = (gisdata['junctions'].loc['733'].geometry.x, gisdata['junctions'].loc['733'].geometry.y)

wn.add_reservoir(resv_name, base_head=base_head, head_pattern=None, coordinates=coordinates)

pipe = wn.get_link('860')
pipe.end_node = wn.get_node('R-1')
wn.remove_node('733')
    
pipe = wn.get_link('914')
pipe.start_node = wn.get_node('R-1')
wn.remove_node('791')

wn.add_pump('TillaHPHSE', '894', '893', pump_type = 'POWER', pump_parameter = 0.1*745.699872, initial_status="OPEN")
wn.remove_link('1011')

#wn.add_pump('PHouse', '427', '426', pump_type = 'POWER', pump_parameter = 0.1*745.699872, initial_status="OPEN")
wn.add_pump('PHouse', '370', '282', pump_type = 'POWER', pump_parameter = 0.1*745.699872, initial_status="OPEN")
wn.remove_link('347')

wn.add_pump('RegalHPHSE', '759', '760', pump_type = 'POWER', pump_parameter = 0.1*745.699872, initial_status="OPEN")
wn.remove_link('889')

wnf = wntr.epanet.io.InpFile()
# wnf.write(box_incore_path + '/Data/' + 'Seaside_dummy_model.inp', wn, units='GPM')
