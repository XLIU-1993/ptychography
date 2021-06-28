from __future__ import division
# pynx
from pynx.wavefront import Wavefront, PropagateNearField, PropagateFarField

# IO
import threading

# Math
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import matplotlib.cm as cm
import matplotlib.patches as mpatches

# Saving files
import os,sys,shutil,csv
from ctypes import windll
import json
import time

from scipy.ndimage import interpolation
'''
in order to perform a more reasonnable simulation of diffraction pattern.
'''
##########################################################################################
# fill in the blacnk
##########################################################################################
# experiment log
experiment_log ='''
This is the first scrit that willed be used to
simulate helix ptycho simulation.
The logics seems to be good.
'''

# path info
path_dir_working = sys.path[0]

# obj info
'''
if only one path was given, it will generate a pure phase obj, in this case leave other path 'None'
'''
obj_path_ampimage = 'D:\\scripts\\20210416_PyNx\\20210528_DongTycho\\Simulation_David\\prototype2_reduite.bmp'
obj_path_phaseimage = 'D:\\scripts\\20210416_PyNx\\20210528_DongTycho\\Simulation_David\\prototype6_2.bmp'
obj_size = (1000e-6,1000e-6) # meter
obj_nearfield = False # True/Flase

# cam info
cam_obj_distance = 10e-2 # meter
cam_pxlsize = 13e-6 # meter
'''
the camera shape will influence the shape of gaussian,
recommend to define an square shape camera, which will be less
problematic.
'''
cam_pxlnb = (512,512)
cam_binning = 1
cam_qe = 0.02
cam_dark_noise = 2.9 # e-
cam_dark_current = 3e-4 # e-/pixe/s
cam_sensitivity = 1
cam_bitdepth = 16
cam_baseline = 100

# probe info
probe_type_list = ['guass'] # do not change
probe_type = probe_type_list[0]  
probe_wavelength = 20e-9 #meter
'''
out of the aradius of probe_sigma_ratio*sigma_pxlnb_min, the intensity will be 
considered as zero, recommend to set the ratio as 2.
as 2D gaussain has two sigma values, the minimum sigma value will be taken
in case of an asymetric gaussian.
'''
probe_sigma_ratio = 1 
probe_max_photonnb =  1e5
probe_bg_photonnb = 1e2

# scan info
scan_type_list = ['rect','spiral'] # do not change
scan_type = scan_type_list[1]
'''
scan_sigma_ratio*sigma_pxlnb will be taken as scan_step_pxlnb, scan_sigma_ratio 
is recommended to set as 1*probe_sigma_ratio, thus we will have a recovering ratio
of 50%.
scan_sigma_ratio/probe_sigma_ratio     recovering ratio
                0                           100%
                2                            0%   
'''
scan_sigma_ratio = 1*probe_sigma_ratio 
scan_nb = 100

##########################################################################################
# define functions
##########################################################################################
def calc_obj_pxlsize(probe_wavelength,cam_obj_distance,cam_pxlnb,cam_pxlsize,nearfield):
    '''
    return a tuple of obj_pxlsize
    '''
    if nearfield:
        obj_xpxlsize = cam_pxlsize
        obj_ypxlsize = cam_pxlsize

    else:
        obj_xpxlsize = probe_wavelength*cam_obj_distance/(cam_pxlnb[0]*cam_pxlsize)
        obj_ypxlsize = probe_wavelength*cam_obj_distance/(cam_pxlnb[1]*cam_pxlsize)
    obj_pxlsize = (obj_xpxlsize,obj_ypxlsize)
    return obj_pxlsize

def calc_obj_pxlnb(obj_size,obj_pxlsize):
    '''
    return a tuple of obj_pxlnb
    '''
    obj_pxlnb = np.array((int(obj_size[0]//obj_pxlsize[0]),
                        int(obj_size[1]//obj_pxlsize[1])))
    return obj_pxlnb

def verify_array_memory(arrayshape,info='unknown'):
    try:
        np.ones(arrayshape)
    except MemoryError:
        print(f'{info} with shape {arrayshape.shape} is too big! simulation stopped!')
        exit()

'''
# instead of simulating a probe with giving sigma value which leads to some overextend problems,
# based on the assumption, that all light from probe matrix should be captured by the camera,
# the properate radius of the probe will be calculated by the pixel size of the object and the 
# pixel number of the camera.

def calc_probe_sigma_pxlnb(probe_sigma_size,obj_pxlsize):
    probe_sigma_pxlnb = np.array((probe_sigma_size[0]//obj_pxlsize,probe_sigma_size[1]//obj_pxlsize))
    print(f'obj_pxlnb:{probe_sigma_pxlnb}')
    return probe_sigma_pxlnb
'''

''' 
# since the probe size will be calculated automatically, the probe size could be extremely small accroding the
# size of the obj pixel and the number of camera pixel. 

def verify_probe_sampling(probe_sigma_pxlnb,probe_sigma_ratio,cam_pxl_nb):
    #1.check yourself if the probe_sigma_pxlnb was too low.
    #2.ensure that intensity over 2sigma is still inside the oxl range.
    if int(2*probe_sigma_ratio*probe_sigma_pxlnb[0]) > cam_pxl_nb or int(2*probe_sigma_ratio*probe_sigma_pxlnb[1]) > cam_pxl_nb:
        print('camera pixels can not fully sample 2sigma of the probe!')
        isYes = input('enter (y) to continue simulation with a truncated gaussian profil, others to stop simulation.')
        if isYes.lower() == 'y':
            pass
        else:    
            quit()
'''

def read_obj_image(obj_path_list):
    '''
    return an list of images
    '''
    obj_image_list = []
    for obj_image_path in obj_path_list:
        if not(obj_image_path) is None:
            try:
                obj_image_list.append(ImageOps.grayscale(Image.open(obj_image_path)))
            except FileNotFoundError:
                print(f'{obj_image_path} is not found, please follow constructions, dont play with me :), simulation stopped!')
    return obj_image_list

def resize_obj_image(obj_image_list,obj_pxlnb):
    '''
    return an list of resized images 
    '''
    obj_imageresize_list = []
    for obj_image in obj_image_list:
        obj_imageresize_list.append(np.asarray(obj_image.resize((obj_pxlnb[0],obj_pxlnb[1]),resample=Image.BICUBIC)))
    return obj_imageresize_list

def make_obj(obj_image_list):
    '''
    return a 2d complexe matrix, amp*exp(ij*phase)
    amp is normalised to 1(equals 1 if no amp image), 
    the phase is strecth from -pi to pi.
    '''
    if len(obj_image_list) == 1:
        amp = 1
        phase = obj_image_list[0]
    else:
        amp = obj_image_list[0]/obj_image_list[0].max()
        phase = obj_image_list[1]
    phase = phase-phase.min()
    phase_range = 2*np.pi
    phase = phase_range*(phase/phase.max()-1/2)
    obj = amp*np.exp(1j*phase)    
    return obj

def pad_obj(obj,cam_pxlnb):
    '''
    return padded image.and obj_pxllim (left,bottom)
    '''
    obj_xpxlnb,obj_ypxlnb = obj.shape
    cam_xpxlnb,cam_ypxlnb = cam_pxlnb
    obj_xpxlnb_pad = obj_xpxlnb + cam_xpxlnb + 4
    obj_ypxlnb_pad = obj_ypxlnb + cam_ypxlnb + 4
    verify_array_memory((obj_xpxlnb_pad,obj_ypxlnb_pad),info='obj_pad')
    obj_pad = np.zeros((obj_xpxlnb_pad,obj_ypxlnb_pad),dtype=np.complex)
    left = cam_xpxlnb//2+2
    bottom = cam_ypxlnb//2+2
    obj_pxllim = left,bottom
    obj_pad[left:left+obj_xpxlnb,bottom:bottom+obj_ypxlnb] = obj[::]
    return obj_pad,obj_pxllim

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

def get_probe_gauss_intensity(probe_efield,probe_max_photonnb=1,probe_bg_photonnb=0):
    """
    return the intensity distribution of a know electric field.
    """
    probe_ifield = probe_efield**2
    return (probe_ifield/probe_ifield.max())*probe_max_photonnb+probe_bg_photonnb

def rect(pas,nb):
    '''
    creat a matrix of the most possible rectangular, return
    the x,y position sucessively.
    '''
    square_longth = int(np.sqrt(nb))
    residu_nb = nb-square_longth**2
    if residu_nb%square_longth == 0 :
        xlongth = square_longth+residu_nb/square_longth
    else:
        xlongth = square_longth+residu_nb//square_longth+1
    yaxis = np.arange(square_longth)
    yaxis_reverse = np.flip(yaxis,0)
    current_nb = 0
    xpos = 0
    xcord = [] 
    ycord = []
    while xpos<xlongth:
        for ypos in yaxis:
            if current_nb<nb:
                xcord.append(xpos)
                ycord.append(ypos)
                current_nb+=1
        xpos+=1        
        for ypos in yaxis_reverse:
           if current_nb<nb:
                xcord.append(xpos)
                ycord.append(ypos)
                current_nb+=1
        xpos+=1
    xcord = np.array(xcord)*pas
    ycord = np.array(ycord)*pas
    xcord = xcord - int(xcord.mean())
    ycord = ycord - int(ycord.mean())
    return xcord,ycord

def spiral_archimedes(pas, nb):
    """" Creates np points spiral of step pas, with pas between successive points
    on the spiral. Returns the x,y coordinates of the spiral points.

    This is an Archimedes spiral. the equation is:
      r=(pas/2*pi)*theta
      the stepsize (radial distance between successive passes) is pas
      the curved absciss is: s(theta)=(pas/2*pi)*integral[t=0->theta](sqrt(1*t**2))dt
    """
    vr, vt = [0], [0]
    t = np.pi
    while len(vr) < nb:
        vt.append(t)
        vr.append(pas * t / (2 * np.pi))
        t += 2 * np.pi / np.sqrt(1 + t ** 2)
    vt, vr = np.array(vt), np.array(vr)
    return np.round(vr * np.cos(vt)).astype(int), np.round(vr * np.sin(vt)).astype(int)

def make_scan(scan_type,scan_step_pxlnb,scan_nb):
    '''
    return a tuple of (xpos,ypos)
    '''
    if scan_type == 'rect':
        posx, posy = rect(scan_step_pxlnb,scan_nb)
    elif scan_type == 'spiral':
        posx, posy = spiral_archimedes(scan_step_pxlnb,scan_nb)
    return posx,posy

def align_scan_obj(scan_position,obj_pxlnb,obj_pxlnb_pad,obj_pxllim):
    '''
    the pynx can reconstruct the obj even if the scanning positions were not given in a sequence,
    but here I need the scanning position to be a continuous sequece so that I can filter out these
    positions where the probe exceeds the obj_pxlnb_pad. Hence the function that generate scan position
    should return the positions in a manner of continious. Otherwise the obj should be scaled to fit
    the scanning position. In this case, there is no necessary to specify the real size of the obj.
    return: a tuple of aligned scan position
    '''
    objxmid = obj_pxlnb_pad[0]//2
    objymid = obj_pxlnb_pad[1]//2

    scanxpos = scan_position[0]
    scanypos = scan_position[1]
    scanxmid = (scanxpos.min()+scanxpos.max())//2
    scanymid = (scanypos.min()+scanypos.max())//2

    xtranslation = objxmid - scanxmid
    ytranslation = objymid - scanymid
    scanxpos = scanxpos+xtranslation
    scanypos = scanypos+ytranslation

    left,bottom = obj_pxllim
    right = left + obj_pxlnb[0]
    top = bottom + obj_pxlnb[1]

    scanxpos_temp = []
    scanypos_temp = []
    for idx in range(len(scanxpos)):
        if right>=scanxpos[idx]>=left and top>=scanypos[idx]>=bottom:
            scanxpos_temp.append(scanxpos[idx])
            scanypos_temp.append(scanypos[idx])
        else:
            print(f'probe exceed obj range at scan num.{idx}, corresponding pixel possion is({scanxpos[idx]},{scanypos[idx]}).')
            print('scan point deleted.')
            break

    if len(scanxpos_temp)<=1:
        print(f'no scan falls into the obj, generate one scan at obj center:{objxmid},{objymid}')
        scanxpos_temp.append(objxmid)
        scanypos_temp.append(objymid)      
    return scanxpos_temp,scanypos_temp

def make_dir(path_dir_working):
    current_time = time.strftime("%Y%m%d%H%M",time.localtime())
    path_dir_newsimulation = path_dir_working+'\\'+current_time+'_ptycho_simulation'
    path_dir_experiment = path_dir_newsimulation+'\\01_experiment_simulation'
    path_dir_diffraction = path_dir_newsimulation+'\\02_diffraction_patterns'
    folder_list = [
                    path_dir_newsimulation,
                    path_dir_experiment,
                    path_dir_diffraction
                    ]
    if os.path.exists(path_dir_newsimulation):
        print('simulation dir exist, clearing...')
        for filename in os.listdir(path_dir_newsimulation):
            file_path = os.path.join(path_dir_newsimulation, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        for folder_path in folder_list:
            os.mkdir(folder_path)
    return path_dir_newsimulation,path_dir_experiment,path_dir_diffraction


def save_fromarray(array,path_dir_experiment,imagetitle):
    '''
    save array as 16bit tiff image
    '''
    path = path_dir_experiment + '\\' + imagetitle + '.tiff'
    pattern = ((2**16-1)/array.max()*(array-array.min())).astype(np.uint16)
    pattern = Image.fromarray(pattern)
    pattern.save(path)

class cameraADconvertor():
    '''
    light: the incoming light source.
    qe: quantum efficiency at corresponding wavelength.
    dark_noise: depends on read_out speed, for andor the dark_noise
                is a combination of AD conversion noise and sensor read
                out noise, measured with single pixel at -80 degree 
                with minimum exposure time. e-
    dark_current: overall average of the sensor. e-/pixel/sec
    binning: bin factor(identical for both side)
    sensitivity: not psecified in andor.
    bitdepth: 16bits
    baseline: not speified in andor.
    '''
    def __init__(self,
                    qe=0.02,
                    dark_noise=2.9,
                    dark_current=3e-4,
                    binning = 1,
                    sensitivity = 1,
                    bitdepth = 16,
                    baseline = 100
                    ):
        self.qe = qe
        self.dark_noise = dark_noise
        self.dark_current = dark_current
        self.binning = binning
        self.sensitivity = sensitivity
        self.bitdepth = bitdepth
        self.baseline = baseline

    def applycamera(self,light):
        self.light = light
        self.generate_poisson()
        self.photonToelectron()
        self.generate_dark_noise()
        self.applybinning()
        self.anologTodigital()
        return(self.light)

    def generate_poisson(self):
        try:
            self.light = np.random.poisson(self.light)
        except ValueError:
            print('Memory error during the generation of photon noise! Simulation stoped!')
            exit()

    def photonToelectron(self):
        self.light = np.round(self.qe*self.light)

    def generate_dark_noise(self):
        '''
        the dark noise will be modeled as a gaussian distribution whose standard deviation is
        equivalent to the dark noise of the camera.
        '''
        self.light = np.round(np.random.normal(
            scale=self.dark_noise,
            size=self.light.shape)+self.light+self.dark_current)
    
    def applybinning(self):
        if self.binning>1:
            newshape = self.light.shape[0]//self.binning, self.binning, self.light.shape[1]//self.binning, self.binning
            self.light.reshape(newshape).sum(-1).sum(1)

    def anologTodigital(self):
        '''
        analog-to-digital units
        '''
        max_depth = np.int(2**self.bitdepth-1)
        self.light = (self.light*self.sensitivity).astype(int)
        self.light += self.baseline
        self.light[self.light>max_depth] = max_depth

def make_diffraction(obj_pad,probe_Efield,scan_position_align):
    obj_xpxlnb,obj_ypxlnb = obj_pad.shape
    probe_shape_xpxlnb,probe_shape_ypxlnb = probe_Efield.shape
    scan_xposition, scan_yposition = scan_position_align
    x0 = scan_xposition-probe_shape_xpxlnb//2
    y0 = scan_yposition-probe_shape_ypxlnb//2
    if x0<0 or y0<0 or x0+probe_shape_xpxlnb>obj_xpxlnb or y0+probe_shape_ypxlnb>obj_ypxlnb:
        print('OVER RANGE at scan position(pixels):',scan_xposition,scan_yposition)
    return probe_Efield*obj_pad[x0:x0+probe_shape_xpxlnb,y0:y0+probe_shape_ypxlnb]

def get_diffration(obj_pad,
                    probe_Efield,
                    scan_position_align,
                    cam,
                    probe_wavelength,
                    obj_pxlsize,
                    obj_nearfield,
                    cam_obj_distance,
                    probe_max_photonnb,
                    probe_bg_photonnb
                    ):
    E_diffracted = make_diffraction(obj_pad,probe_Efield,scan_position_align)
    E_wavefront = Wavefront(d=np.fft.fftshift(E_diffracted), 
                            wavelength=probe_wavelength,
                            pixel_size=obj_pxlsize)
    if obj_nearfield:
        E_wavefront = PropagateNearField(dz=cam_obj_distance) * E_wavefront
    else:
        E_wavefront = PropagateFarField(dz=cam_obj_distance) * E_wavefront
    # taking photon nb and background
    I_wavefront = abs(E_wavefront.get(shift=True))**2*probe_max_photonnb+probe_bg_photonnb
    # problem of dimension
    I_wavefront = np.squeeze(I_wavefront,axis=0)
    I_wavefront = cam.applycamera(I_wavefront)
    return I_wavefront

def save_patterns(path,intensity_temp,name):
    file = path+'\\'+str(name)+'.tiff'
    imagei = Image.fromarray(obj=intensity_temp.astype(np.uint16),
                            mode='I;16')
    imagei.save(file,'tiff')
##########################################################################################
# premairy calculations and verification
##########################################################################################
# obj
'''
calculate obj_pxlsize and estimate obj_pxlnb, ensure the matrix won't exceed the max length
of np.array
'''
obj_pxlsize = calc_obj_pxlsize(probe_wavelength,cam_obj_distance,cam_pxlnb,cam_pxlsize,obj_nearfield) #meter
obj_pxlnb = calc_obj_pxlnb(obj_size,obj_pxlsize)
obj_size = obj_pxlsize[0]*obj_pxlnb[0],obj_pxlsize[1]*obj_pxlnb[1]
print(f'obj_pxlsize:{obj_pxlsize[0]:.3e},{obj_pxlsize[1]:.3e} meter.')
print(f'obj_pxlnb:{obj_pxlnb}.')
print(f'obj_size:{obj_size[0]:3e},{obj_size[1]:3e} meter.')
verify_array_memory(obj_pxlnb,info='obj before padding')

'''
read and resize obj images acrodding to the obj_pxlnb and make the obj
'''
obj_path_list = [obj_path_ampimage,obj_path_phaseimage]
obj_image_list = read_obj_image(obj_path_list)
obj_imageresize_list = resize_obj_image(obj_image_list,obj_pxlnb)
obj = make_obj(obj_imageresize_list)

'''
pad the obj
'''
obj_pad,obj_pxllim = pad_obj(obj,cam_pxlnb)
obj_pxlnb_pad = obj_pad.shape
obj_size_pad = obj_pxlsize[0]*obj_pxlnb_pad[0],obj_pxlsize[1]*obj_pxlnb_pad[1]
print(f'obj_pxlnb_pad:{obj_pxlnb_pad}.')
print(f'obj_size_pad:{obj_size_pad[0]:3e},{obj_size_pad[1]:3e} meter.')

# probe
'''
calculate probe information
'''
probe_shape_pxlnb = cam_pxlnb
probe_sigma_pxlnb = probe_shape_pxlnb[0]//(2*probe_sigma_ratio),probe_shape_pxlnb[1]//(2*probe_sigma_ratio)
probe_sigma_size = probe_sigma_pxlnb[0]*obj_pxlsize[0],probe_sigma_pxlnb[1]*obj_pxlsize[1]
print(f'probe_shape_pxlnb:{probe_shape_pxlnb}.')
print(f'probe_sigma_pxlnb:{probe_sigma_pxlnb}.')
print(f'probe_sigma_size is:{probe_sigma_size[0]:.3e},{probe_sigma_size[1]:.3e} meter.')

'''
make probe
'''
probe_Efield = make_probe_gauss(probe_shape_pxlnb,probe_sigma_ratio)
probe_Ifield = get_probe_gauss_intensity(probe_Efield,probe_max_photonnb,probe_bg_photonnb)
probe_bg = np.ones((probe_Ifield.shape))*probe_bg_photonnb

# scan
scan_step_pxlnb = np.array(probe_sigma_pxlnb).min()*scan_sigma_ratio
scan_step_size = scan_step_pxlnb*obj_pxlsize
print('scan_step_pxlnb',scan_step_pxlnb)
print(f'scan_step_size:{scan_step_size:.3e} meter.')

if scan_step_pxlnb >= np.array(obj_pxlnb).min():
    print('obj is smaller than one scan step')
scan_position = make_scan(scan_type,scan_step_pxlnb,scan_nb)
scan_position_align = align_scan_obj(scan_position,obj_pxlnb,obj_pxlnb_pad,obj_pxllim)

# creat dir
path_dir_newsimulation,path_dir_experiment,path_dir_diffraction = make_dir(path_dir_working)

# show obj
fig1 = plt.figure(tight_layout=True)

ax11 = fig1.add_subplot(121)
ax11.set_title('amplitude image')
ax11.set_xlabel('meter')
ax11.set_ylabel('meter')
ax11.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
amp_array =np.abs(obj)
ax11.imshow(amp_array,
            extent=[0,obj_pxlnb[0]*obj_pxlsize[0],0,obj_pxlnb[1]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

ax12 = fig1.add_subplot(122)
ax12.set_title('phase image')
ax12.set_xlabel('meter')
ax12.set_ylabel('meter')
ax12.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
phase_array = np.angle(obj)
ax12.imshow(phase_array,
            extent=[0,obj_pxlnb[0]*obj_pxlsize[0],0,obj_pxlnb[1]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

path_obj_image = path_dir_experiment + '\\obj_image.tiff'
fig1.savefig(path_obj_image)
plt.show()

# show probe
fig2 = plt.figure(tight_layout=True)

ax21 = fig2.add_subplot(121)
ax21.set_title('probe linear')
ax21.set_xlabel('meter')
ax21.set_ylabel('meter')
ax21.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
ax21.imshow(probe_Ifield,
            extent=[0,probe_shape_pxlnb[0]*obj_pxlsize[0],0,probe_shape_pxlnb[1]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

ax22 = fig2.add_subplot(122)
ax22.set_title('probe log10')
ax22.set_xlabel('meter')
ax22.set_ylabel('meter')
ax22.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
ax22.imshow(np.log10(probe_Ifield),
            extent=[0,probe_shape_pxlnb[0]*obj_pxlsize[0],0,probe_shape_pxlnb[1]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

path_probe_image = path_dir_experiment + '\\probe_image.tiff'
fig2.savefig(path_probe_image)
plt.show()

# show scan
fig3 = plt.figure(tight_layout=True)
grid = fig3.add_gridspec(2,3)

# init obj as scan position background
ax31 = fig3.add_subplot(grid[0,0])
ax31.set_title('scan position')
ax31.set_xlabel('meter')
ax31.set_ylabel('meter')
ax31.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))

ax31.imshow(np.abs(obj_pad),
            extent=[0,obj_pxlnb_pad[0]*obj_pxlsize[0],0,obj_pxlnb_pad[1]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

# init obj as scan area background
ax32 = fig3.add_subplot(grid[1,0])
ax32.set_title('scan area')
ax32.set_xlabel('meter')
ax32.set_ylabel('meter')
ax32.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))

ax32.imshow(np.abs(obj_pad),
            extent=[0,obj_pxlnb_pad[0]*obj_pxlsize[0],0,obj_pxlnb_pad[1]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

# init camera background as diffraction pattern background
ax33 = fig3.add_subplot(grid[:,1:])
ax33.set_title('background')
ax33.set_xlabel('meter')
ax33.set_ylabel('meter')
ax33.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))

dict_camADconverotr ={
    'qe':cam_qe,
    'dark_noise':cam_dark_noise,
    'dark_current':cam_dark_current,
    'binning':cam_binning,
    'sensitivity':cam_sensitivity,
    'bitdepth':cam_bitdepth,
    'baseline':cam_baseline,
}
cam = cameraADconvertor(**dict_camADconverotr)
cam_bg = cam.applycamera(light=probe_bg)
save_patterns(path_dir_experiment,cam_bg,'cam_bg')

image33 = ax33.imshow(cam_bg,
                    extent=[0,obj_pxlnb_pad[0]*obj_pxlsize[0],0,obj_pxlnb_pad[1]*obj_pxlsize[1]],
                    cmap='Greys_r',
                    interpolation='nearest')
plt.pause(2)

# scan nb
new_scan_nb = len(scan_position_align[0])
# scan position in meter with respect to obj
scan_xposition_real = np.array(scan_position_align[0])*obj_pxlsize[0]
scan_yposition_real = np.array(scan_position_align[1])*obj_pxlsize[1]
# displaying colors
colors = cm.rainbow(np.linspace(0, 1, new_scan_nb))
# contour of the illumination area
circ_radius = np.array(probe_sigma_size).min()*probe_sigma_ratio

for idx in range(new_scan_nb):
    # animation of scan position
    xi = scan_xposition_real[idx]
    yi = scan_yposition_real[idx]
    ci = colors[idx]
    ax31.scatter(xi,yi,color=ci)

    # animation of scan area
    circ_i = mpatches.Circle((xi,yi),radius=circ_radius,color=ci,fill=False)
    ax32.add_patch(circ_i)

    # animation of diffraction pattern
    scan_iposition_align = scan_position_align[0][idx],scan_position_align[1][idx]
    intensity_temp = get_diffration(obj_pad,
                                probe_Efield,
                                scan_iposition_align,
                                cam,
                                probe_wavelength,
                                obj_pxlsize[0], # not support different size
                                obj_nearfield,
                                cam_obj_distance,
                                probe_max_photonnb,
                                probe_bg_photonnb
                                )
    ax33.set_title('scan'+str(idx))
    image33.set_data(intensity_temp)
    # saving diffration pattern
    save_patterns(path_dir_diffraction,intensity_temp,idx)
    plt.pause(1)

path_scan_image = path_dir_experiment + '\\scan_image.tiff'
fig3.savefig(path_scan_image)

plt.pause(2)

dict_extrainfo={
    'extra_path_workingdir':path_dir_working,
}

dict_objinfo={
    'obj_path_ampimage':obj_path_ampimage,
    'obj_path_phaseimage':obj_path_phaseimage,
    'obj_size':obj_size, # meter np.array((None,None))
    'obj_pxlsize':obj_pxlsize, # meter
    'obj_nearfield':obj_nearfield, 
}

dict_objinfo_sup={
    'obj_pxlsize':obj_pxlsize, # meter
    'obj_pxlnb':obj_pxlnb, # np.array((None,None))
    'obj_size':obj_size,
    'obj_pxlnb_pad':obj_pxlnb_pad,
    'obj_size_pad':obj_size_pad,
}

dict_caminfo={
    'cam_obj_distance':cam_obj_distance, # meter
    'cam_pxlsize':cam_pxlsize,
    'cam_pxlnb':cam_pxlnb, # equal size at both side
    'cam_binning':cam_binning,
    'cam_qe':cam_qe,
    'cam_dark_noise':cam_dark_noise,
    'cam_dark_current':cam_dark_current,
    'cam_sensitivity':cam_sensitivity,
    'cam_bitdepth':cam_bitdepth,
    'cam_baseline':cam_baseline, 
}

dict_probeinfo={
    'probe_type_list':probe_type_list, # list
    'probe_type':probe_type, # str
    'probe_sigma_ratio':probe_sigma_ratio,
    'probe_max_photonnb':probe_max_photonnb,
    'probe_bg_photonnb':probe_bg_photonnb,
    'probe_wavelength':probe_wavelength, # meter
}

dict_probeinfo_sup={
    'probe_shape_pxlnb':probe_shape_pxlnb, # idem camera_pxlnb
    'probe_sigma_pxlnb':probe_sigma_pxlnb, # np.array((None,None))
    'probe_sigma_size':probe_sigma_size, # meter np.array((None,None))
}

dict_scaninfo={
    'scan_type_list':scan_type_list, # list of str ['rect','spiral']
    'scan_type':scan_type, # str
    'scan_sigma_ratio':scan_sigma_ratio, 
    'scan_nb':new_scan_nb,
}

dict_scaninfo_sup={
    'scan_step_pxlnb':scan_step_pxlnb,
    'scan_step_size':scan_step_size, # meter
}

dict_info={
    'extra_info':dict_extrainfo,
    'obj_info':dict_objinfo,
    'obj_info_sup':dict_objinfo_sup,
    'probe_info':dict_probeinfo,
    'probe_info_sup':dict_probeinfo_sup,
    'scan_info':dict_scaninfo,
    'scan_info_sup':dict_scaninfo_sup,
    'cam_info':dict_caminfo
}