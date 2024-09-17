import wntr
import geopandas as gpd
import numpy as np
import random
import pandas as pd
try:
    import rasterio
    from osgeo import gdal
except ImportError:  # pragma: no cover
    rasterio = gdal = None
from pathlib import Path
from hashlib import sha1
#%% Functions
def _query_raster(nodes, filepath, band):
    """
    Query a raster for values at coordinates in a DataFrame's x/y columns.

    Parameters
    ----------
    nodes : pandas.DataFrame
        DataFrame indexed by node ID and with two columns: x and y
    filepath : string or pathlib.Path
        path to the raster file or VRT to query
    band : int
        which raster band to query

    Returns
    -------
    nodes_values : zip
        zipped node IDs and corresponding raster values
    """
    # must open raster file here: cannot pickle it to pass in multiprocessing
    with rasterio.open(filepath) as raster:
        values = np.array(
            tuple(raster.sample(nodes.values, band)), dtype=float
        ).squeeze()
        values[values == raster.nodata] = np.nan
        return zip(nodes.index, values)
    
def get_node_elevations_raster(junctions, filepath, band=1):
    """Add `elevation` attribute to each node from local raster file(s).

    If `filepath` is a list of paths, this will generate a virtual raster
    composed of the files at those paths as an intermediate step.

    Args:
        junctions (pandas.DataFrame): Function expects x and y columns for coordinates
        filepath (_type_): _description_
        band (int, optional): _description_. Defaults to 1.

    Raises:
        ImportError: _description_

    Returns:
        elevs: (pandas.Series)
    """

    if rasterio is None or gdal is None:  # pragma: no cover
        raise ImportError("gdal and rasterio must be installed to query raster files")

    # if a list of filepaths is passed, compose them all as a virtual raster
    # use the sha1 hash of the filepaths list as the vrt filename
    if not isinstance(filepath, (str, Path)):
        filepaths = [str(p) for p in filepath]
        sha = sha1(str(filepaths).encode("utf-8")).hexdigest()
        filepath = f"./.osmnx_{sha}.vrt"
        gdal.BuildVRT(filepath, filepaths).FlushCache()

    elevs = pd.Series(dict(_query_raster(junctions, filepath, band)))

    return elevs

def find_next_name(wn_reg,naming_convention):
    existing_item_names = [item for item in wn_reg]
    i = 1
    while True:
        if f'{naming_convention}{i}' not in existing_item_names:
            return f'{naming_convention}{i}'
        else:
            i +=1
            
def add_pump(wn,pump_name,start_node_name,pipe_name,direction='N'):
    drct = {'N':(0,5),'E':(5,0),'S':(0,-5),'W':(-5,0)}
    start_node = wn.get_node(start_node_name)
    new_node_name = find_next_name(wn.nodes,'J-')
    wn.add_junction(new_node_name,base_demand=0,demand_pattern='1',elevation=start_node.elevation,
                    coordinates=(start_node.coordinates[0]+drct[direction][0],
                                 start_node.coordinates[1]+drct[direction][1]))
    wn.add_pump(pump_name,start_node_name,new_node_name) # Add pump from start node to new node
    # Fix pipe
    pipe = wn.get_link(pipe_name)
    if pipe.start_node_name == start_node_name:
        pipe.start_node = wn.get_node(new_node_name) # Assign new starting node
    elif pipe.end_node_name == start_node_name:
        pipe.end_node = wn.get_node(new_node_name) # Assign new ending node
    else:
        print('Error. Pipe provided not connected to node provided')
    return wn.get_link(pump_name)

#%% Load in GIS files
if __name__ == "__main__":
    folder_path = r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\My Documents\GitHub\Seaside_WDN_Control\data"
    crs = 'EPSG:32610' # UTM Zone 10N for Oregon, code EPSG:32610 in meters

    bldg_pop_path = f'{folder_path}/seaside_bldg_pop.geojson'
    bldg_pop = gpd.read_file(bldg_pop_path)

def seaside_bldg_pop_adjustment(bldg_pop_og, crs, tourist_population = 3000):
    # Create copies to not change original dataframes from user
    bldg_pop = bldg_pop_og.copy()
    
    bldg_pop.to_crs(crs,inplace=True)
    # Cleaning data from building geojson file
    keep_columns = ['struct_typ','year_built','no_stories','appr_bldg','dgn_lvl','guid','origin',
                    'rmv_improv','rmv_land','elev','strctid','numprec','ownershp','race','hispan',
                    'vacancy','gqtype','livetype','landuse','geometry']
    bldg_pop = bldg_pop[keep_columns]
    # Fix units for the following columns: string to float
    fix_units = ['numprec','ownershp','race','hispan','vacancy','gqtype']
    for col in fix_units:
        bldg_pop[col] = bldg_pop[col].replace('',np.nan)
        bldg_pop[col] = bldg_pop[col].astype(float) #6645 pop, now 7234.
    
    # Various settings
    # 6645 residents as assigned currently, commercial not occupied
    commercial_occupied = False # assume night time, everyone at home
    commercial_resident_pop_percent = 0.4 # Percentage of people not at home and at commercial buildings
    
    if commercial_occupied:
        # Assign some residential population to commercial buildings
        print('Not implemented')
        # Assign some tourist population to commercial buildings
        
        # Assign tourist population to rentals
        
    else: # Assume commercial bldgs are empty (night?)
        # Assign tourist population to rentals
        low_rental_min, low_rental_max = 1,5
        high_rental_min, high_rental_max = 10,40
        t_list = []
        rental_bldgs = bldg_pop[bldg_pop['vacancy'] == 5].copy()
        shuffled_rentals = rental_bldgs.sample(frac=1).reset_index(drop=True).copy()
        # set index as guid
        bldg_pop.set_index('guid', inplace=True)
        shuffled_rentals.set_index('guid', inplace=True)
        bldg_guid_pop = {}
        for guid, row in shuffled_rentals.iterrows():
            if row['landuse'] == 'losr':
                if bldg_pop.loc[guid, 'numprec'] == 0:
                    people = random.randint(low_rental_min, low_rental_max)
                    bldg_guid_pop[guid] = people
                    tourist_population -= people
                    t_list.append(tourist_population)
            elif row['landuse'] == 'hosr':
                if bldg_pop.loc[guid, 'numprec'] == 0:
                    people = random.randint(high_rental_min, high_rental_max)
                    bldg_guid_pop[guid] = people
                    tourist_population -= people
                    t_list.append(tourist_population)
            else:
                assert row['landuse'] in ['losr','hosr'], 'Landuse must be losr or hosr for seasonal rentals'
            if tourist_population <= 0:
                #print('Finished assigning tourist population to seasonal rentals')
                break
        if tourist_population > 0:
            #print('Not enough tourists added on initial pass, looping again')
            while tourist_population > 0:
                for guid, row in shuffled_rentals.iterrows():
                    # Add more people to the high occupancy rentals
                    if row['landuse'] == 'hosr':
                        if row['guid'] in bldg_guid_pop:
                            existing_people = bldg_guid_pop[guid]
                        else: # had population already assigned to it preciously
                            existing_people = bldg_pop.loc[guid, 'numprec']
                        people = random.randint(low_rental_min, low_rental_max)
                        bldg_guid_pop[guid] = people  + existing_people
                        tourist_population -= people
                        t_list.append(tourist_population)
                        # Check if done adding tourist population
                    if tourist_population <= 0:
                        #print('Finished assigning tourist population to seasonal rentals')
                        break
        
    
    for index_value, population in bldg_guid_pop.items():
        if index_value in bldg_pop.index:
            bldg_pop.at[index_value, 'numprec'] = population
        else:
            print('error')
    print(f'Total pop: {sum(list(bldg_pop["numprec"]))}, Tourists added: {sum(list(bldg_guid_pop.values()))}')
    
    return bldg_pop

#%% FUNCITON FOR CREATING SEASIDE WN WITH BLDG POP
if __name__ == "__main__":
    # Run function
    bldg_pop_adj = seaside_bldg_pop_adjustment(bldg_pop, crs, tourist_population = 3000)
    
    # Add shapefile paths
    pipe_shpfile = f'{folder_path}/Seaside_water_pipelines_wgs84.shp'
    node_shpfile = f'{folder_path}/Seaside_wter_nodes.shp'
    list_dem_paths = [f'{folder_path}/seaside_elevation_raster.tif']
    
    # Load shapefiles
    water_pipelines = gpd.read_file(pipe_shpfile)
    water_nodes = gpd.read_file(node_shpfile)


def create_seaside_wn(bldg_pop_og, water_pipelines_og, water_nodes_og, list_dem_paths, crs):
    # Create copies to not change original dataframes from user
    water_pipelines = water_pipelines_og.copy()
    water_nodes = water_nodes_og.copy()
    bldg_pop = bldg_pop_og.copy()
    
    water_pipelines.to_crs(crs,inplace=True)
    bldg_pop.to_crs(crs,inplace=True)
    # Get elevation for nodes
    water_nodes.to_crs("epsg:4326", inplace=True)
    water_nodes["x"] = water_nodes.apply(lambda row : row.geometry.coords[0][0],axis=1)
    water_nodes["y"] = water_nodes.apply(lambda row : row.geometry.coords[0][1],axis=1)
    water_nodes["elevation"] = get_node_elevations_raster(water_nodes.loc[:,["x","y"]], list_dem_paths, band=1)
    water_nodes = water_nodes.drop(columns=["x", "y"])
    water_nodes.to_crs(crs, inplace=True)
    
    # Fix water pipeline information
    pipe_to_fix = water_pipelines[water_pipelines['Link_ID'] == 470]
    if water_pipelines.loc[pipe_to_fix.index[0], 'tonode'] != 344:
        water_pipelines.loc[pipe_to_fix.index[0], 'tonode'] = 344
    
    ### With building population assigned, now snap buildings to pipes and add population to nodes
    tolerance = 500
    snap_buildings = wntr.gis.snap(bldg_pop, water_pipelines, tolerance)
    snap_buildings['node'] = None
    # Get all buildings nodes that are closer to the from node and assign the node index to them
    temp = snap_buildings['line_position'] < 0.5
    snap_buildings.loc[temp, 'node'] = water_pipelines.loc[snap_buildings.loc[temp, 'link'], 'fromnode'].values
    # Get all buildings nodes that are closer to the to node and assign the node index to them
    temp = snap_buildings['line_position'] >= 0.5
    snap_buildings.loc[temp, 'node'] = water_pipelines.loc[snap_buildings.loc[temp, 'link'], 'tonode'].values
    # Set population of water nodes to 0 and sum up population for all buildings assigned to that node
    water_nodes['node_id_'] = water_nodes['node_id']
    water_nodes.set_index('node_id',inplace=True)
    water_nodes['pop'] = 0
    for node_id, group_df in snap_buildings.groupby('node'):
        # print(f'Node ID: {node_id}')
        # print(group_df.head())
        total_pop = bldg_pop.loc[group_df.index,'numprec'].sum() 
        water_nodes.loc[node_id,'pop'] = total_pop
    #print('Lost population due to nan: ', sum(list(water_nodes[water_nodes['guid'].isna()]['pop'])))
    water_nodes.dropna(subset=['guid'], inplace=True)
    water_nodes['demand_gpm'] = water_nodes['pop'] * 80 / (24*60)  
    
    ### Information needed for creating the EPAnet model
    
    material_roughness_dict = {'TJDCI':140 , # Roughness dictionary
                               'CIP':130 ,
                               'TJCI':110 ,
                               'DCI':140 ,
                               'STEEL':150 ,
                               'MJCI':110 ,
                               'TRANSITE':140 , # Asbestos cement
                               'PLASTIC BLUE BRUTE':150 ,
                               'STEEL ID':100 ,
                               'PVC':150 ,
                               'TTE':130 , # unknown
                               'STEEL OD':100 ,
                               'TJCP':135 , # Copper?
                               'CI':110 ,
                               'TRANSITE & PLASTIC':145 ,
                               'DCIP':140 ,
                               'HDPE':140 ,
                               'PLASTIC':140 ,
                               'CONC':120 ,
                               'ID STEEL':100 ,
                               'BELL & HUB':110 , # cast iron
                               'OD STEEL':100 ,
                               'ID':130 } # unknown
    
    # Some diameters are missing from the shapefile, must add manually here
    missing_diameters_dict = {'P-48':2,
                              'P-6':12,
                              'P-184':4,
                              'P-392':4,
                              'P-2':12,
                              'P-4':12,
                              'P-496':12,}
    
    ### Create model!
    wn = wntr.network.WaterNetworkModel()
    
    # Add all junctions from shapefile
    for index, row in water_nodes.iterrows():
        node_id = f"J-{int(row['node_id_'])}"  # Assuming 'node_id' is the column name for node IDs
        coordinates = (row['geometry'].x, row['geometry'].y)
        elevation = row['elevation'] 
        demand = row['demand_gpm'] / 15850.323141488905 # GPM to m^3/ sec
        wn.add_junction(node_id, base_demand=demand, demand_pattern='1', elevation=elevation,
        coordinates=coordinates)
        
    # Add reservoir manually
    wn.add_reservoir('R-1', base_head=215*0.3048, coordinates=(428576.16275396466, 5089379.424101184))
    
    # Add pipelines connecting junctions
    for index, row in water_pipelines.iterrows():
        pipe_id = f"P-{int(row['Link_ID'])}"
        from_node = f"J-{int(row['fromnode'])}"
        to_node = f"J-{int(row['tonode'])}"
        length = row['length_km'] * 1000 # to meters
        if pipe_id in missing_diameters_dict: # Add missing diameters from dictionary
            diameter = missing_diameters_dict[pipe_id] * 0.0254 # inches to meters
        else:
            diameter = row['diameter'] * 0.0254 # inches to meters
        if row['Pipe_type'] in material_roughness_dict:
            roughness = material_roughness_dict[row['Pipe_type']]
        else:
            roughness = 130
        wn.add_pipe(pipe_id, from_node, to_node, length=length, diameter=diameter, roughness=roughness,initial_status='OPEN')
     
    # Missing pipes connecting reservoir to network
    next_pipe_name = find_next_name(wn.links,'P-')
    wn.add_pipe(next_pipe_name, 'R-1', 'J-178', length=1, diameter=12*0.0254, roughness=130,initial_status='OPEN')
    next_pipe_name = find_next_name(wn.links,'P-')
    wn.add_pipe(next_pipe_name, 'R-1', 'J-179', length=1, diameter=12*0.0254, roughness=130,initial_status='OPEN')
    
    # Add and connect in pumps to the network
    phouse_pump = add_pump(wn,'PHouse','J-373','P-220','E') # Pipe 166 Pump House do Pipe 220 instead? nodes 352 to 365, instead nodes 373 to 389
    regal_elk_pump = add_pump(wn,'RegalHPHSE','J-415','P-463','E') # Pipe 463 Regal Elk Pump HSE nodes 415 to 416
    tillamook_pump = add_pump(wn,'TillaHPHSE','J-3','P-487','W') # Pipe 487 Tillamook Head Pump HSE nodes 3 to 1
    wn.get_link('RegalHPHSE').power = 3.67 * 1.5 #regal 0.005 HP, 3.677 W
    wn.get_link('PHouse').power = 220.65 * 1.5 #phouse 0.3 HP, 220.65 W
    wn.get_link('TillaHPHSE').power = 110.325 * 1.5# tillamook 0.15 HP, 110.325 W
    
    # obj = wntr.epanet.InpFile()
    # obj.write(f'my_seaside.inp',wn)
    
    return wn

if __name__ == "__main__":
    # Run the function
    water_network = create_seaside_wn(bldg_pop_adj, water_pipelines, water_nodes, list_dem_paths, crs)