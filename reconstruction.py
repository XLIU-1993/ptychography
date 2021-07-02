from __future__ import division

# ptycho
from pynx.ptycho import *

# read save files
import os,sys,json


# math
import numpy as np

##########################################################################################
# import simulationd data
##########################################################################################
# give simulation directory
path_dir_simulation = 'D:\\scripts\\ptychography\\202106290934_ptycho_simulation'
# auto fill directory
path_dir_experiment = path_dir_simulation+'\\simulation_info'
path_dir_diffraction = path_dir_simulation+'\\diffraction_patterns'
# auto fill path
path_simulation_info = path_dir_experiment+'\\simulation_info.txt'
path_scan_position = path_dir_experiment+'\\scan_position.csv'
path_cam_bg = path_dir_experiment+'\\cam_bg.tiff'

# read simulation info
with open(path_simulation_info, 'r') as f: 
    dict_simulationinfo = json.load(f)

# extract simulation for reconstruction
cam_obj_distance = dict_simulationinfo['cam_info']['cam_obj_distance']
cam_pxlsize = dict_simulationinfo['cam_info']['cam_pxlsize']
probe_wavelength = dict_simulationinfo['probe_info']['probe_wavelength']
print(f'cam_obj_distance:{cam_obj_distance} meter.')
print(f'cam_pxlsize:{cam_pxlsize} meter.')
print(f'probe_wavelength:{probe_wavelength} meter.')

# extract simulation for innitial guess
obj_pxlsize = dict_simulationinfo['obj_info_sup']['obj_pxlsize']
cam_pxlnb = dict_simulationinfo['cam_info']['cam_pxlnb']

print(f'obj_pxlsize:{obj_pxlsize} meter.')
print(f'cam_pxlnb:{cam_pxlnb} .')
##########################################################################################
# define functions
##########################################################################################
def make_probe_gauss(probe_shape_pxlnb,probe_sigma_ratio,center=(0,0)):
    """
    return a circularly masked 2D gaussian electric field distribution without rotation, centered at (0,0)
    """
    probe_sigma_pxlnb = probe_shape_pxlnb[0]//(2*probe_sigma_ratio),probe_shape_pxlnb[1]//(2*probe_sigma_ratio)
    nx, ny = probe_shape_pxlnb
    v = np.array(probe_sigma_pxlnb) ** 2

    x = np.linspace(-nx//2, nx//2, nx)
    y = np.linspace(-ny//2, ny//2, ny)
    xx, yy = np.meshgrid(x, y)

    g = np.exp(-((xx-center[0])**2/(v[0])+(yy-center[1])**2/(v[1])))
    return g

##########################################################################################
# initial guess
##########################################################################################
