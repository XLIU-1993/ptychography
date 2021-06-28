from __future__ import division

# ptycho
from pynx.ptycho import *
import numpy as np

##########################################################################################
# import simulationd data
##########################################################################################
path_dir_simulation = 'G:\PYNX\ptychography\202106290443_ptycho_simulation'
path_dir_experiment = path_dir_simulation+'\\simulation_info'
path_dir_diffraction = path_dir_simulation+'\\diffraction_patterns'

https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
Of course, to read the json back you can use this: with open(path, 'r') as f: data = json.load(f) , which returns a dictionary with your data. – tsveti_iko Aug 20 '18 at 7:06 
That's for reading the json file and then to deserialize it's output you can use this: data = json.loads(data) – tsveti_iko Aug 20 '18 at 7:17