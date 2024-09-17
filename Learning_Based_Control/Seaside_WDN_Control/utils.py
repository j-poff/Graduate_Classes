import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
# rasterio and gdal are optional dependencies for raster querying
try:
    import rasterio
    from osgeo import gdal
except ImportError:  # pragma: no cover
    rasterio = gdal = None
from pathlib import Path
from hashlib import sha1
from shapely.geometry import Point
import pandas as pd

import numpy as np
import contextily as ctx
# gpd.options.use_pygeos = True

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

def process_line_data(lines, threshold):
    '''
    given a lines DataFrame and a threshhold number, modifies the lines to include the start and end coords of each line,

    '''
    #takes transmission/water lines 
    #how those lines connect to neighboring lines
    print('### Processing line data ###')
    ### Create start and end nodes for each line
    nodes = []
    geometry = []
    lines['start_node'] = None #create new columns
    lines['end_node'] = None
    j = 0
    for i, line in lines.iterrows(): #loop through every line
        try:
            geom = line.geometry.geoms[0]
        except:
            geom = line.geometry
        
        geometry.append(Point([geom.coords[0][0], geom.coords[0][1]])) 
        nodes.append({'Line': i, 'Node': j}) #node list holds the line ID
        #unique number for each start node and end node for each line 
        lines.loc[i,'start_node'] = j
        j = j+1
            
        geometry.append(Point([geom.coords[-1][0], geom.coords[-1][1]]))
        nodes.append({'Line': i, 'Node': j})
        lines.loc[i,'end_node'] = j
        j = j+1
            
    nodes = gpd.GeoDataFrame(nodes, geometry=geometry)
    nodes.set_crs(lines.crs, inplace=True)
        
    ### Group nodes into supernodes
    #supernode is composed of all nodes within certain radius 
    nodes_buffer = nodes.copy()
    nodes_buffer.geometry = nodes_buffer.buffer(threshold) #how big is the buffer, to connect lines 
    intersect_nodes = wntr.gis.intersect(nodes, nodes_buffer)

    intersect_nodes['visited'] = False #new column
    nodes['supernode'] = None
    #walking through each node, and if node is visited, don't need to assign to supernode again
    #con: specific to order of walking
    for i in intersect_nodes.index:
        if intersect_nodes.loc[i,'visited'] == True:
            continue
        for j in intersect_nodes.loc[i,'intersections']: #list of intersecting vertices
            if intersect_nodes.loc[j,'visited'] == True: #mybe see which point is closer, the current one of the new one
                continue
            nodes.loc[j, 'supernode'] = i
            intersect_nodes.loc[j,'visited'] = True

    assert intersect_nodes['visited'].all()

    ### Updates lines dataframe with start and end supernode columns
    map_node_to_supernode = nodes['supernode']
    lines['start_supernode'] = map_node_to_supernode[lines['start_node']].values
    lines['end_supernode'] = map_node_to_supernode[lines['end_node']].values
    
    return nodes, lines


def find_connected_components(G):
    uG = G.to_undirected()

    ### Find connected components
    cc = nx.connected_components(uG)
    cc_node_attr = {}
    for i, component in enumerate(cc):
        for node in component:
            cc_node_attr[node] = i

    cc = nx.connected_components(uG)
    largest_cc = max(cc, key=len)
    
    return cc_node_attr, largest_cc

def remove_unconnected_components(wn):
    G = wn.to_graph()
    cc_node_attr, largest_cc = find_connected_components(G)

    # Plot connected components
    ax = wntr.graphics.plot_network(wn, node_attribute=cc_node_attr,
                                    node_size=40, node_cmap='tab20')
    # Plot largest connected component
    ax = wntr.graphics.plot_network(wn, node_attribute=list(largest_cc),
                                    node_size=10)
    print('Percent nodes connected within 1m', len(largest_cc)/wn.num_nodes)

    ### Remove unconnected junctions and pipes
    print(wn.describe(1))
    unconnected = set(wn.node_name_list) - largest_cc
    for uncon in unconnected:
        for uncon_link in wn.get_links_for_node(uncon):
            wn.remove_link(uncon_link)
        wn.remove_node(uncon)

    print(wn.describe(1))
    
    return wn
    

def load_data(service_area, data_dir, crs = "EPSG:32161"):
    ### Load PRGOV data
    print('### LOADING DATA ###')
    try:
        ### Load local data files after running the script for the first time
        mains = gpd.read_file(service_area+"_mains.geojson")
        wtps = gpd.read_file(service_area+"_wtps.geojson")
        tanks = gpd.read_file(service_area+"_tanks.geojson")
        pumps = gpd.read_file(service_area+"_pumps.geojson")
        nodes = gpd.read_file(service_area+"_nodes.geojson")
        buildings = gpd.read_file(service_area+"_buildings.geojson")
    except:
        water_dir = data_dir+"/Infrastructure/PRGOV_Water"
        mains = gpd.read_file(water_dir+"/w_Main.geojson")
        mains = mains.loc[mains['OperArea'] == service_area,:]
        mains.to_crs(crs, inplace=True)
        
        mask = mains.dissolve().convex_hull
        
        wtps = gpd.read_file(water_dir+"/w_Treatment_Plant.geojson")
        #if service_area == 'SJN':
        #    wtps = wtps.loc[(wtps['OperArea'] == service_area) | (wtps['Name'] == 'Los Filtros Guaynabo') ,:]
        #else:
        #    wtps = wtps.loc[wtps['OperArea'] == service_area,:]
        wtps.to_crs(crs, inplace=True)
        wtps = wtps.clip(mask)
        
        wtp_capacity = wtps['CapacityMGD'].sum() # MGD
        wtp_capacity = (wtp_capacity*1e6)/(24*3600*264.17) # m3/s
        
        pumps = gpd.read_file(water_dir+"/w_Pump_Station.geojson")
        pumps = pumps.loc[pumps['OperArea'] == service_area,:]
        pumps.to_crs(crs, inplace=True)
        pumps = pumps.clip(mask)
        
        tanks = gpd.read_file(water_dir+"/w_Storage_Tank.geojson")
        tanks = tanks.loc[tanks['OperArea'] == service_area,:]
        tanks.to_crs(crs, inplace=True)
        tanks = tanks.clip(mask)
        
        buildings = gpd.read_file(data_dir+"/Infrastructure/HDX_Buildings/hotosm_pri_buildings_polygons.shp")
        buildings.to_crs(crs, inplace=True)
        buildings = buildings.clip(mask)
        
        fig, ax = plt.subplots()
        mains.plot(ax=ax, color='b')
        wtps.plot(ax=ax, color='r')
        pumps.plot(ax=ax, color='k')
        tanks.plot(ax=ax, color='g')
        
        mains.to_file(service_area+"_mains.geojson", driver='GeoJSON')
        wtps.to_file(service_area+"_wtps.geojson", driver='GeoJSON')
        tanks.to_file(service_area+"_tanks.geojson", driver='GeoJSON')
        pumps.to_file(service_area+"_pumps.geojson", driver='GeoJSON')
        nodes.to_file(service_area+"_nodes.geojson", driver='GeoJSON')
        buildings.to_file(service_area+"_buildings.geojson", driver='GeoJSON')

    return mains, wtps, pumps, tanks, buildings

def find_connected_components(G):
    uG = G.to_undirected()

    ### Find connected components
    cc = nx.connected_components(uG)
    cc_node_attr = {}
    for i, component in enumerate(cc):
        for node in component:
            cc_node_attr[node] = i

    cc = nx.connected_components(uG)
    largest_cc = max(cc, key=len)
    
    return cc_node_attr, largest_cc

def remove_unconnected_components(wn):
    G = wn.to_graph()
    cc_node_attr, largest_cc = find_connected_components(G)

    # Plot connected components
    ax = wntr.graphics.plot_network(wn, node_attribute=cc_node_attr,
                                    node_size=40, node_cmap='tab20')
    # Plot largest connected component
    ax = wntr.graphics.plot_network(wn, node_attribute=list(largest_cc),
                                    node_size=10)
    print('Percent nodes connected within 1m', len(largest_cc)/wn.num_nodes)

    ### Remove unconnected junctions and pipes
    print(wn.describe(1))
    unconnected = set(wn.node_name_list) - largest_cc
    for uncon in unconnected:
        for uncon_link in wn.get_links_for_node(uncon):
            wn.remove_link(uncon_link)
        wn.remove_node(uncon)

    print(wn.describe(1))
    
    return wn

def water_demand_generation(water_pipelines, buildings, census_blocks, list_dem_paths, crs, tourist_pop = 0, 
                            gal_per_person = 80, plot=True, building_snap_tolerance=100):
    '''

    Parameters
    ----------
    water_pipelines : geodataframe
        Line data for the pipelines.
    buildings : geodataframe
        Builing polygon data for area .
    census_blocks : geodataframe
        Population field = 'POP20'.
    list_dem_paths : list
        paths to DEM sources.
    crs : str
        ESPG CRS desired for data.
    tourist_pop : int, optional
        Number of tourists expected in buildings in area. The default is 0.
    gal_per_person : float, optional
        Number of gallons per person per day. The default is 80.
    plot : bool, optional
        Plot figures through process. The default is True.
    building_snap_tolerance : float, optional
        In units of CRS, the furtherst distance buildings can be considered associated with node. The default is 100 meters.

    Returns
    -------
    nodes : geodataframe
        Nodes with supernode information.
    base_demand : TYPE
        For each supernode in m^3 / second.

    '''
    
    water_pipelines = water_pipelines.to_crs(crs)
    assert water_pipelines.crs == crs
    # water_nodes = water_nodes.to_crs(crs)
    buildings = buildings.to_crs(crs)
    assert buildings.crs == crs
    buildings['area_sqm'] = buildings.geometry.area
    buildings = buildings[buildings['area_sqm'] != 0] # elimate the weird point data

    #Plot
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        water_pipelines.plot(ax=ax, color='blue', label='Water Pipelines')
        # water_nodes.plot(ax=ax, color='red', label='Nodes')
        buildings.plot(ax=ax,color='black', label = 'Buildings')
        ctx.add_basemap(ax, zoom='auto',crs=crs,source=ctx.providers.OpenStreetMap.Mapnik)
        ax.legend()
        plt.show()

    # Process water lines into separate lines and nodes
    dist_threshold = 10
    nodes, lines = process_line_data(water_pipelines, threshold=dist_threshold)
    if plot:
        fig, ax = plt.subplots()
        lines.plot(ax=ax, color='b')
        nodes.plot(ax=ax, color='red')

    # Mask
    mask = water_pipelines.dissolve().convex_hull

    ## Add elevation, using a DEM
    nodes.to_crs("epsg:4326", inplace=True)
    nodes["x"] = nodes.apply(lambda row : row.geometry.coords[0][0],axis=1)
    nodes["y"] = nodes.apply(lambda row : row.geometry.coords[0][1],axis=1)
    nodes["elevation"] = get_node_elevations_raster(nodes.loc[:,["x","y"]], list_dem_paths, band=1)
    nodes = nodes.drop(columns=["x", "y"])
    nodes.to_crs(crs, inplace=True)
    assert nodes.crs == crs
    # nodes["elev_diff"] = nodes.apply(lambda row : abs(row.elevation - row._raster_elevation),axis=1)

    ### Assign buildings to nearest end point to lines and compute total building area
    building_centers = buildings.centroid.to_frame('geometry')
    building_centers['area'] = buildings.area
    tolerance = building_snap_tolerance  # m

    snap_buildings = wntr.gis.snap(building_centers, lines, tolerance)
    snap_buildings['node'] = None
    temp = snap_buildings['line_position'] < 0.5
    snap_buildings.loc[temp, 'node'] = lines.loc[snap_buildings.loc[temp, 'link'], 'start_supernode'].values
    temp = snap_buildings['line_position'] >= 0.5
    snap_buildings.loc[temp, 'node'] = lines.loc[snap_buildings.loc[temp, 'link'], 'end_supernode'].values
    nodes['total_building_area'] = 0
    for i, group in snap_buildings.groupby('node'):
        total_building_area = building_centers.loc[group.index,'area'].sum()
        nodes.loc[i,'total_building_area'] = total_building_area

    if plot:
        fig, ax = plt.subplots()
        water_pipelines.plot(ax=ax, color='b')
        building_centers.plot(ax=ax, color='red')
        building_centers.loc[snap_buildings.index,:].plot(ax=ax, color='green')


    
    census_blocks.to_crs(crs, inplace=True)
    cb = census_blocks.clip(mask)
    resident_pop = cb['POP20'].sum()

    print('Resident population', resident_pop)
    
    total_population = tourist_pop + resident_pop
    total_gal_per_day = total_population * gal_per_person
    wtp_capacity = total_gal_per_day/ (3600*24*264.17) #convert to m^3 / sec from gal / day
    print('Gallons of water per person per day', (wtp_capacity*3600*24*264.17)/total_population)
    # print('Gallons of water per person per day', (total_gal_per_day/total_population))


    supernodes = nodes['supernode'].unique()

    # Demand estimate (very basic)
    #num_supernodes = len(supernodes) # some of these later get removed (if not connected)
    #base_demand = wtp_capacity/num_supernodes

    # Demand estimate (based on fraction of building area)
    supernode_building_area = nodes.groupby('supernode')['total_building_area'].sum()
    upper_bound = supernode_building_area.quantile(0.95)
    supernode_building_area[supernode_building_area > upper_bound] = upper_bound # removes outliers
    supernode_building_fraction = supernode_building_area/supernode_building_area.sum()
    base_demand = wtp_capacity*supernode_building_fraction
    base_demand.index = base_demand.index.astype(str)

    return nodes, base_demand