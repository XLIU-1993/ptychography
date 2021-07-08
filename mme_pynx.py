'''
this is a minimum executable exmple(MEE) of ptychography 
simulation based on pynx version-2020.02.02.
                                          xliu@imagine-optic.com
'''
from numpy.core.shape_base import block
from pynx2019.ptycho import *
from pynx2019.ptycho import simulation,shape
import numpy as np

'''
IMPORTANT:
1.To view cxi. data need to install slix package, and type following command in terminal:
  silx view 'file_path'
2.The overflow problem of PyNx:
  Replace the corresponding lines in 'ptycho.py' of the function 'calc_regularisation_scale(self):'
    probe_size = self._probe[0].size*1.0
    obj_size = self._obj[0].size*1.0
    data_size = self.data.iobs.size*1.0
    nb_photons = self.nb_obs*1.0
3.All important parameters are below, it's basiclly what you need to change to 
  perform a customized Ptycho simulation. 
'''

# Path of the amplitude and the phase
ampIm_path = 'D:\\scripts\\20210416_PyNx\\20210528_DongTycho\\Simulation_David\\prototype2_reduite.bmp'
phaseIm_path = 'D:\\scripts\\20210416_PyNx\\20210528_DongTycho\\Simulation_David\\prototype6_2.bmp'

wavelength = 420e-9 # (meter)

# Camera
cam_pxl_nb = 256  # detector pixel number
cam_pxl_size =  52e-6 # size of a single pixel of the camera (meter)
cam_obj_distance = 339e-6 # distance between camera and object(meter)

# Object
obj_pxl_nb = 256 # object pixel number
obj_pxl_size = wavelength*cam_obj_distance/cam_pxl_size/cam_pxl_nb # size of single pixel in the object plane (meter)

# Probe
probe_size = 4.492e-07 # Probe radius at 1/e^2 (meter)
scan_step_size = 5.391e-07 # Scan step (meter)
nb_scan = 200 # Number of scan

# Algorithm
nb_iteration = 100
use_DM = True
use_AP = False
use_ML = False
ExportData = False


'''
IMPORTANT:
Save a copy before change content below
'''
######################################################
# Import the image of phase object:
######################################################
print('************************************************')
print('           SIMULATION of an EXPERIMENT          ')
print('************************************************')
print(f'wavelength(µm):{wavelength*10**6:.3f},camera pixel size(µm):{cam_pxl_size*10**6:.3f},camera size(µm):{cam_pxl_size*cam_pxl_nb*10**6:.3f}')
print(f'camera object distance(mm):{cam_obj_distance*10**3:.3f},object pixel size(µm):{obj_pxl_size*10**6:.3f},object size(µm):{obj_pxl_size*obj_pxl_nb*10**6:.3f}')

import matplotlib.pyplot as plt
from PIL import Image,ImageOps

if ExportData:
    import os,sys
    import time
    working_path = sys.path[0]
    current_time = time.strftime("%Y%m%d%H%M",time.localtime())
    cxi_folder_path = working_path+'\\'+current_time+'_PTYCHO_simulation'

amp_origin =  Image.open(ampIm_path)
phase_rbg = Image.open(phaseIm_path)
phase_gray = ImageOps.grayscale(phase_rbg)
phase_origin = phase_gray

fig1 = plt.figure('construct an real object')
ax1 = fig1.add_subplot(231)
ax1.set_title(f'Origin ampitude image:{amp_origin.size}')
ax1.imshow(amp_origin)

ax2 = fig1.add_subplot(234)
ax2.set_title(f'Origin phase image:{phase_origin.size}')
ax2.imshow(phase_origin)

amp_resize = amp_origin.resize((obj_pxl_nb,obj_pxl_nb),resample=Image.BICUBIC)
phase_resize = phase_origin.resize((obj_pxl_nb,obj_pxl_nb),resample=Image.BICUBIC)
# Image.resize(size, resample=0, box=None) https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters

ax3 = fig1.add_subplot(232)
ax3.set_title(f'Resized ampitude image:{amp_resize.size}')
ax3.imshow(amp_resize)

ax4 = fig1.add_subplot(235)
ax4.set_title(f'Resized phase image:{phase_resize.size}')
ax4.imshow(phase_resize)

amp_resize = np.array(amp_resize)
phase_resize = np.array(phase_resize)

#Cunstruct an real image:
fft_ampIm = np.fft.fft2(amp_resize)
fft_phaseIm = np.fft.fft2(phase_resize)
fft_obj = np.multiply(np.abs(fft_ampIm),np.exp(1j*np.angle(fft_phaseIm)))

obj_complexe = np.fft.ifft2(fft_obj)
obj_pxl_nb = int(obj_complexe.shape[0])
obj_amp = np.real(obj_complexe).reshape(obj_pxl_nb,obj_pxl_nb)

ax5 = fig1.add_subplot(133)
ax5.set_title(f'Image of reconstructed object:{obj_amp.shape}')
ax5.imshow(obj_amp)

plt.tight_layout()
plt.show()

if ExportData:
    if os.path.exists(cxi_folder_path):
        pass
    else:
        os.mkdir(cxi_folder_path)
    #ax3Im = ax3.get_tightbbox(fig1.canvas.get_renderer(),for_layout_only=True).transformed(fig1.dpi_scale_trans.inverted())
    ax3Im = ax3.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    ax4Im = ax4.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    ax5Im = ax5.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    fig1.savefig(cxi_folder_path+'\\ampitude_image.png', bbox_inches=ax3Im)
    fig1.savefig(cxi_folder_path+'\\phase_image.png', bbox_inches=ax4Im)
    fig1.savefig(cxi_folder_path+'\\objet_image.png', bbox_inches=ax5Im)

######################################################
# Geometries of the illumination
###################################################### 
probe_omega_pxl_nb = int(probe_size/obj_pxl_size)
probe_shape = (cam_pxl_nb,cam_pxl_nb)
print(f'probe_omega_pixel_nb:{probe_omega_pxl_nb:.3f},probe shape:{probe_shape}')

######################################################
# Geometries of scan
######################################################
scan_step_pxl_nb = int(scan_step_size/obj_pxl_size)
print(f'scan step size(micron):{scan_step_size*10**6:.3f},scan step pixel number:{scan_step_pxl_nb}')

######################################################
# initial informations of the simulation of an experiment:
######################################################
obj_info = {'type': 'ampl_phase', 'phase_stretch': 1.57, 'alpha_win':.2}
probe_info = {'type':'gauss','shape':probe_shape,'sigma_pix':(probe_omega_pxl_nb, probe_omega_pxl_nb)}
scan_pattern = None
scan_info = {'type': 'spiral', 'scan_step_pix': scan_step_pxl_nb, 'n_scans': nb_scan}
data_info = {'nb_photons_per_frame': 1e9,'bg': 0,'noise':'poisson','wavelength': wavelength,'detector_distance': cam_obj_distance,
             'detector_pixel_size': cam_pxl_size} 

print('************************************************')
print('            SIMULATING EXPERIMENT.......        ')
print('************************************************')
######################################################
# Simulate the dataset of an experiment:
######################################################
data_simulation = simulation.Simulation(obj=obj_complexe,
                                        obj_info=obj_info,
                                        probe_info=probe_info,
                                        scan=scan_pattern,
                                        scan_info=scan_info,
                                        data_info=data_info)
data_simulation.make_data()

scan_positions_x,scan_positions_y = data_simulation.scan.values

fig2 =plt.figure()
ax21 = fig2.add_subplot()
ax21.set_title(f'Scan pattern')
ax21.scatter(scan_positions_x,scan_positions_y)
plt.show()
if ExportData:
  ax21Im = ax21.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
  fig2.savefig(cxi_folder_path+'\\scan-pattern.png', bbox_inches=ax21Im)

# Get amplitude of the 'measured' diffraction pattern
measured_amplitude = data_simulation.amplitude.values

print('************************************************')
print(' Initial Guess of Probe Obj of Ptycho Algorithms ')
print('************************************************')
######################################################
# Prepare initial guess of object and probe for Ptycho algorithm:
######################################################
Reobj_ypxl_nb, Reobj_xpxl_nb = shape.calc_obj_shape(scan_positions_x,scan_positions_y,probe_shape=probe_shape)
print(f'The shape probe is:{probe_shape}')
print(f'The pixel numbers of the reconstructed object is:{Reobj_ypxl_nb}*{Reobj_xpxl_nb}')
print(f'The pixel numbers of the object was:{obj_pxl_nb}*{obj_pxl_nb}')

obj_guess_info = {'type':'random','range':(0.9,1.e3,0,0.5),'shape':(Reobj_ypxl_nb, Reobj_xpxl_nb)}
probe_guess_info = {'type':'gauss','shape':(cam_pxl_nb,cam_pxl_nb),'sigma_pix':(probe_omega_pxl_nb, probe_omega_pxl_nb)} 
basic_data_info = {'wavelength': wavelength, 'detector_distance': cam_obj_distance,'detector_pixel_size': cam_pxl_size}

init_guess = simulation.Simulation(obj_info=obj_guess_info,probe_info=probe_guess_info,data_info=basic_data_info)
init_guess.make_obj()
init_guess.make_probe()

######################################################
# Create Ptycho algorithm Data and Objects
######################################################
print('create ptycho data...')
ptycho_data = PtychoData(iobs=measured_amplitude**2,positions=(scan_positions_x*obj_pxl_size,scan_positions_y*obj_pxl_size),
                        detector_distance=cam_obj_distance,mask=None,pixel_size_detector=cam_pxl_size,wavelength=wavelength)
print('create ptycho object...')
ptycho_object = Ptycho(probe=init_guess.probe.values, obj=init_guess.obj.values, data=ptycho_data, background=None)
print('scaling object probe...')
ptycho_object = ScaleObjProbe(verbose=True)*ptycho_object


print('************************************************')
print('      Start ITERATIVE Ptycho Algorithme         ')
print('************************************************')

if use_DM:
  ptycho_object = DM(update_object=True, update_probe=False, calc_llk=2, show_obj_probe=2)**nb_iteration*ptycho_object
if use_AP:
  ptycho_object = AP(update_object=True, update_probe=False, calc_llk=2, show_obj_probe=2)**nb_iteration*ptycho_object
if use_ML:
  ptycho_object = ML(update_object=True, update_probe=False, calc_llk=2, show_obj_probe=2)**nb_iteration*ptycho_object

######################################################
# Use DM and ML options to smooth the object and/or probe
######################################################
'''
http://ftp.esrf.fr/pub/scisoft/PyNX/doc/tutorial/ptycho_operators.html
ToDO
'''

######################################################
# Add probe modes and continue optimising
######################################################
'''
http://ftp.esrf.fr/pub/scisoft/PyNX/doc/tutorial/ptycho_operators.html
ToDO
'''

######################################################
# Export data and/or result object & probe to CXI (hdf5) files
######################################################
print('************************************************')
print('                Export CXI Data                 ')
print('************************************************')

if ExportData:
    cxi_file_path1 = cxi_folder_path+'\\'+current_time+'_probeobj.cxi'
    cxi_file_path2 = cxi_folder_path+'\\'+current_time+'_data.cxi'
    ptycho_object.save_obj_probe_cxi(cxi_file_path1)
    save_ptycho_data_cxi(cxi_file_path2, measured_amplitude**2, cam_pxl_size, wavelength, cam_obj_distance,
                            scan_positions_x*obj_pxl_size, scan_positions_y*obj_pxl_size,z=None,monitor=None,
                            mask=None,instrument='Simulation', overwrite=True)
print('************************************************')
print('                    FINISHED                    ')
print('************************************************')
