# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:48:56 2023

@author: riversam
"""

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
client = IncoreClient()
data_service = DataService(client)

#%% Readign the building inventory for the city of Seaside. 
bldg_dataset_id = "613ba5ef5d3b1d6461e8c415"        # defining building dataset (GIS point layer)
bldg_dataset = Dataset.from_data_service(bldg_dataset_id, data_service)
#geoviz.plot_map(bldg_dataset, column='struct_typ',category='True')
bldg_df = bldg_dataset.get_dataframe_from_shapefile()
bldg_df.set_index('guid', inplace=True)
print('Number of buildings: {}' .format(len(bldg_df)))


#%% Reading the water infrastrcuture: water pipes, a water treatment plant, and three water pumping stations. 
wter_pipe_dataset_id = "60e72f9fd3c92a78c89636c7"        # defining water pipes (GIS point layer)
wter_pipe_dataset = Dataset.from_data_service(wter_pipe_dataset_id, data_service)
#geoviz.plot_map(wter_pipe_dataset, column=None, category=False)

wter_fclty_dataset_id = "60e5e91960b3f41243faa3b2"        # defining water facilities (GIS point layer)
wter_fclty_dataset = Dataset.from_data_service(wter_fclty_dataset_id, data_service)
#geoviz.plot_map(wter_fclty_dataset, column='utilfcltyc', category=True)