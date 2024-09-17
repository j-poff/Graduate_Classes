import wntr
import tkinter as tk
from tkinter import filedialog
import networkx as nx
import numpy as np
import pandas as pd
import math
import json
import os

#%%
def get_user_input():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.withdraw()
    return file_path

print('Select EPANET file')
epa_inp_file_path = get_user_input()

wnet = wntr.network.WaterNetworkModel(epa_inp_file_path)
wnet_skel, skel_map = wntr.morph.skel.skeletonize(wnet, 36*.0254, branch_trim=True, 
                            series_pipe_merge=True, parallel_pipe_merge=True, 
                            max_cycles=None, use_epanet=True, return_map=True, 
                            return_copy=True)
#os.chdir(r'//depot.engr.oregonstate.edu/users/poffja/Windows.Documents/My Documents/GitHub/Seaside_WDN_Control/src')
os.chdir(r"\\depot.engr.oregonstate.edu\users\poffja\Windows.Documents\Desktop")
wntr.network.write_inpfile(wnet_skel, 'ky1_36_nobranch.inp')
