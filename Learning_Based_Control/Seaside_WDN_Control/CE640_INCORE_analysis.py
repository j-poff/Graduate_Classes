from pyincore import IncoreClient, Dataset, DataService, HazardService, FragilityService
from pyincore import FragilityCurveSet, MappingSet
from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.pipelinedamage import PipelineDamage
from pyincore.analyses.waterfacilitydamage import WaterFacilityDamage
from pyincore.analyses.pipelinedamagerepairrate import PipelineDamageRepairRate
from pyincore.analyses.housingunitallocation import HousingUnitAllocation
from pyincore.analyses.cumulativebuildingdamage import CumulativeBuildingDamage
import os

#%% INCORE ANALYSIS
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
rt = [100, 250, 500, 2500, 5000, 10000]
# rt = [1000]

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
    os.makedirs(path_to_output)
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
        
