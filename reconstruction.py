from __future__ import division

# ptycho
from pynx2019.ptycho import*
from pynx2019.ptycho import shape

# read save files
import os,sys,json
import csv
from PIL import Image

# math
import numpy as np

##########################################################################################
# define functions
##########################################################################################
def make_probe_gauss(probe_shape_pxlnb,probe_sigma_ratio,center=(0,0)):
    """
    return a circularly masked 2D gaussian electric field distribution without rotation, centered at (0,0)
    identical with the gauss function in simulation of didffraction pattern.
    """
    probe_sigma_pxlnb = probe_shape_pxlnb[0]//(2*probe_sigma_ratio[0]),probe_shape_pxlnb[1]//(2*probe_sigma_ratio[1])
    nx, ny = probe_shape_pxlnb
    v = np.array(probe_sigma_pxlnb) ** 2

    x = np.linspace(-nx//2, nx//2, nx)
    y = np.linspace(-ny//2, ny//2, ny)
    xx, yy = np.meshgrid(x, y)

    g = np.exp(-((xx-center[0])**2/(v[0])+(yy-center[1])**2/(v[1])))
    return g

def make_random_obj(obj_pxlnb):
    '''
    return a random object with normalised amplitude to 1 and stretched phase
    from -pi to pi.
    '''
    rand_phase = np.random.uniform(-np.pi,np.pi,obj_pxlnb)
    obj = np.random.uniform(0,1,obj_pxlnb) * np.exp(1j * rand_phase)    
    return obj

def shift_scan(scan_position):
    '''
    return scan position centered at (0,0)
    '''
    posx = np.array(scan_position[0]).astype(np.float)
    posy = np.array(scan_position[1]).astype(np.float)
    meanx = posx.mean()
    meany = posy.mean()
    shiftx = posx - meanx
    shifty = posy - meany
    return (shiftx,shifty)

def get_scan(path_scan_position):
    '''
    return scan_position centered at (0,0).
    '''
    posx = []
    posy = []
    with open(path_scan_position,'r') as f:
        scan_reader = csv.reader(f)
        for row in scan_reader:
            posx.append(row[0])
            posy.append(row[1])
    scan_position = shift_scan((posx,posy))
    return scan_position

def get_diffraction_patterns(path_dir_diffraction):
    '''
    return 3D array of diffraction patterns.
    '''
    images = []
    valid_images = [".tif",".tiff",".bmp"]
    for f in os.listdir(path_dir_diffraction):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(np.array(Image.open(os.path.join(path_dir_diffraction,f))))
    return images
##########################################################################################
# import simulationd data
##########################################################################################
# give simulation directory
path_dir_simulation = 'D:\\scripts\\ptychography\\202107051320_ptycho_simulation'
# auto fill directory
path_dir_experiment = path_dir_simulation+'\\simulation_info'
path_dir_diffraction = path_dir_simulation+'\\diffraction_patterns'
# auto fill path
path_simulation_info = path_dir_experiment+'\\simulation_info.txt'
path_scan_position = path_dir_experiment+'\\scan_position.csv'
path_cam_bg = path_dir_experiment+'\\cam_bg.tiff'

# read simulation info
with open(path_simulation_info,'r') as f: 
    dict_simulationinfo = json.load(f)

# read scan position info
scan_position = get_scan(path_scan_position)
print('scqn nb',len(scan_position[0]))

# read diffraction pattern
intensity = get_diffration
##########################################################################################
# set parameters for known cam_obj_distance
##########################################################################################
# extract simulation for reconstruction
cam_pxlnb = dict_simulationinfo['cam_info']['cam_pxlnb']
cam_pxlsize = dict_simulationinfo['cam_info']['cam_pxlsize']
probe_wavelength = dict_simulationinfo['probe_info']['probe_wavelength']
cam_obj_distance = dict_simulationinfo['cam_info']['cam_obj_distance']

print(f'cam_pxlnb:{cam_pxlnb} .')
print(f'cam_pxlsize:{cam_pxlsize} meter.')
print(f'probe_wavelength:{probe_wavelength} meter.')
print(f'cam_obj_distance:{cam_obj_distance} meter.')

# probe
'''
out of the radius of probe_sigma_ratio*probe_shape_pxlnb, 
the intensity will be considered as zero, recommend to set the ratio as 
(2,2), since gauss intensity distribution outside 2sigma is consider as 0.
'''
probe_shape_pxlnb = cam_pxlnb
probe_sigma_ratio = (2,2)  # DEFINE

# obj
 


##########################################################################################
# initial guess
##########################################################################################
# probe
probe_Efield = make_probe_gauss(probe_shape_pxlnb,probe_sigma_ratio)

# obj
obj_pxlsize = dict_simulationinfo['obj_info_sup']['obj_pxlsize']
print(f'obj_pxlsize:{obj_pxlsize} meter.')
#obj_pxlnb = shape.calc_obj_shape(posx/pixel_size_object, posy/pixel_size_object, ampl.shape[1:])