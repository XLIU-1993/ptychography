from __future__ import division
from operator import add
# pynx
from pynx.wavefront import Wavefront, PropagateNearField, PropagateFarField

# IO
import threading

# Math
import numpy as np

# Matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import matplotlib.cm as cm
import matplotlib.patches as mpatches

# Saving files
import os,sys,shutil,csv
from ctypes import windll
import json
import time

'''
in the process of developing a gui for pynx, i found the simulation
is not realistic, one can not simulate a real situation.
in order to perform a more realistic simulation of diffraction pattern,
some constrains are given here:
firstly, i assume the camera takes the entire
diffraction pattern originates from the probe which must have the same
pixel numbers as the camera does.
acrroding to the first assumption, the numerical aperture of the diffraction
pattern is fixed by the distance between oobject and camera, and the size
of the camera.
one can hereby deduce, neither in far field or near field, the pixel size of
the object, the pixel size of the object is the same for the one of the probe,
then, the matrix that represents the probe has a certain form with a related
real size.
considering the fact that the divergence of the diffraction pattern is unknown, the
distance between camera and object was set by one's experiences, we could
specify a probe, like give a certain sigma size of a guassian distribution of 
electric field. but it will leads to complecated calculations for the scanning
pattern and we need to see how will the probe exceed the limited probe matrix.
thus, in stead of giving a real size of the probe, only a ratio of the sigma
pixel numbers which will be confined inside the probe matrix, will be 
necessarily specified by the user.
the ratio works good with the simulation, but i only did it for guassian.
farthur more, in real experimence, the perfermence of camera is fatal, i added
a virtual camera which has real parameters of an Andor M series camera.
The effect of saturation will be clearly observed. Expect the case where given 
a too much photon number will lead to valueError of np.poisson(). 
simulation of diffraction pattern with a non symetric shape of camera is possible.
'''
##########################################################################################
# fill in the blacnk
##########################################################################################
# experiment log
txt_readme ='''
This is the first scrit that willed be used to
simulate helix ptycho simulation.
The logics seems to be good.
'''

# path info
path_dir_working = sys.path[0]

# obj info
'''
if only one path was given, it will generate a pure phase obj,
in this case leave other path as 'None'
'''
obj_path_ampimage = 'D:\\scripts\\20210416_PyNx\\20210528_DongTycho\\Simulation_David\\prototype2_reduite.bmp'
obj_path_phaseimage = 'D:\\scripts\\20210416_PyNx\\20210528_DongTycho\\Simulation_David\\prototype6_2.bmp'
#obj_path_ampimage = 'G:\\PYNX\\Test\\sample_obj.tif'
#obj_path_phaseimage = 'G:\\PYNX\\Test\\sample_phase.jpg'
obj_size = (18e-6,13e-6) # meter (xsize,ysize)
obj_nearfield = False # True/Flase

# cam info
cam_obj_distance = 2.028e-3 # meter
cam_pxlsize = 52e-6 # meter
'''
the camera shape will influence the shape of gaussian,
recommend to define an square shape camera, which will be less
problematic.
'''
cam_pxlnb = (256,256) # (row_nb,column_nb)
cam_binning = 1
cam_qe = 0.02
cam_dark_noise = 2.9 # e-
cam_dark_current = 3e-4 # e-/pixe/s
cam_sensitivity = 1
cam_bitdepth = 16
cam_baseline = 50

# probe info
probe_type_list = ['guass'] # do not change
probe_type = probe_type_list[0]  
probe_wavelength = 420e-9 #meter
'''
out of the radius of probe_sigma_ratio*probe_shape_pxlnb, 
the intensity will be considered as zero, recommend to set the ratio as 
(2,2), since gauss intensity distribution outside 2sigma is consider as 0.

as long as the camera has an even shape, the probe_sigma_ratio should be 
identical for 2 sides, otherwise, to make a symetric gauss, the ratio should
be calculated by considering the ratio of the camera shape.
'''
probe_sigma_ratio = (5,5) #(x_ratio,y_ratio)
probe_max_photonnb =  1e7
probe_bg_photonnb = 20

# scan info
scan_type_list = ['rect','spiral'] # do not change
scan_type = scan_type_list[0]
'''
scan_recover_ratio is defined as the recovering ratio of the two
matrix used to present the probe.
if scan_recover_ratio is 0, it means that the two matrix are just
not superposing. 
if the scan_recover_ratio is 1, it means that the two matrix are
on top of each other.
scan_sigma_ratio*sigma_pxlnb.min() will be taken as scan_step_pxlnb.
    scan_recover_ratio          scan_sigma_ratio
            0                 2*probe_sigma_ratio.min()
            0.5               1*probe_sigma_ratio.min()
            1                 0*probe_sigma_ratio.min()
'''
scan_recover_ratio = 0.92 # 0-1
scan_sigma_ratio = (1-scan_recover_ratio)/0.5*np.array(probe_sigma_ratio).min()
scan_nb = 100

##########################################################################################
# define functions
##########################################################################################
def calc_obj_pxlsize(probe_wavelength,cam_obj_distance,cam_pxlnb,cam_pxlsize,nearfield):
    '''
    return a tuple of obj_pxlsize(xpxlsize,ypixelsize)
    '''
    if nearfield:
        obj_xpxlsize = cam_pxlsize
        obj_ypxlsize = cam_pxlsize
    else:
        obj_xpxlsize = probe_wavelength*cam_obj_distance/(cam_pxlnb[1]*cam_pxlsize)
        obj_ypxlsize = probe_wavelength*cam_obj_distance/(cam_pxlnb[0]*cam_pxlsize)
    obj_pxlsize = (obj_xpxlsize,obj_ypxlsize)
    return obj_pxlsize

def calc_obj_pxlnb(obj_size,obj_pxlsize):
    '''
    return a tuple of obj_pxlnb (row_nb,column_nb)
    '''
    obj_pxlnb = np.array((int(obj_size[1]//obj_pxlsize[1]),
                        int(obj_size[0]//obj_pxlsize[0])))
    return obj_pxlnb

def verify_array_memory(arrayshape,info='unknown'):
    try:
        np.ones(arrayshape)
    except MemoryError:
        print(f'{info} with shape {arrayshape.shape} is too big! simulation stopped!')
        exit()

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
                print(f'{obj_image_path} is not found, simulation stopped!')
    return obj_image_list

def resize_obj_image(obj_image_list,obj_pxlnb):
    '''
    return an list of resized images, resize(column,row)
    '''
    obj_imageresize_list = []
    for obj_image in obj_image_list:
        obj_imageresize_list.append(np.asarray(obj_image.resize((obj_pxlnb[1],obj_pxlnb[0]),resample=Image.NEAREST)))
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
        obj_image_list[0] = obj_image_list[0]-obj_image_list[0].min()
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
    left = cam_ypxlnb//2+2
    bottom = cam_xpxlnb//2+2
    obj_pxllim = left,bottom
    obj_pad[bottom:bottom+obj_xpxlnb,left:left+obj_ypxlnb] = obj[::]
    return obj_pad,obj_pxllim

def make_probe_gauss(probe_shape_pxlnb,probe_sigma_ratio,center=(0,0)):
    """
    return a circularly masked 2D gaussian electric field distribution without rotation, centered at (0,0)
    """
    probe_sigma_pxlnb = probe_shape_pxlnb[1]//(2*probe_sigma_ratio[0]),probe_shape_pxlnb[0]//(2*probe_sigma_ratio[1])
    row, column = probe_shape_pxlnb
    v = np.array(probe_sigma_pxlnb) ** 2

    x = np.linspace(-column//2, column//2, column)
    y = np.linspace(-row//2, row//2, row)
    xx, yy = np.meshgrid(x, y)

    g = np.exp(-((xx-center[0])**2/(v[0])+(yy-center[1])**2/(v[1])))
    return g

def get_probe_gauss_intensity(probe_efield,probe_max_photonnb=1,probe_bg_photonnb=0):
    """
    return the intensity distribution of a know electric field.
    """
    probe_ifield = probe_efield**2
    return (probe_ifield/probe_ifield.max())*probe_max_photonnb+probe_bg_photonnb

def rect(scan_step_pxlnb,scan_nb,obj_pxlnb):
    '''
    creat a matrix of the most possible rectangular, return
    the x,y positions sucessively. centered at 0.
    '''
    ylength = obj_pxlnb[0] # row
    xlength = obj_pxlnb[1] # column
    total_pxl = xlength*ylength # total number
    total_scanpxl = scan_nb*scan_step_pxlnb**2
    if total_scanpxl >  total_pxl:
        total_scanpxl = total_pxl
    
    scan_nb = int(total_scanpxl/scan_step_pxlnb**2)
    ratio = np.sqrt(total_scanpxl/total_pxl)
    scan_rownb = int(ylength*ratio)
    scan_column = int(xlength*ratio)

    current_nb = 0
    xpos = 0 
    ypos = 0
    xcord = [] 
    ycord = []
    while xpos*scan_step_pxlnb<=scan_column:
        while ypos*scan_step_pxlnb <=scan_rownb:
            if current_nb<scan_nb:
                xcord.append(xpos*scan_step_pxlnb)
                ycord.append(ypos*scan_step_pxlnb)
                current_nb+=1
                ypos+=1
            else:
                break
        ypos-=1
        xpos+=1        
        while ypos*scan_step_pxlnb >=0:
            if current_nb<scan_nb:
                xcord.append(xpos*scan_step_pxlnb)
                ycord.append(ypos*scan_step_pxlnb)
                current_nb+=1
                ypos-=1
            else:
                break
        ypos+=1
        xpos+=1
    xcord = np.array(xcord) - int(np.array(xcord).mean())
    ycord = np.array(ycord) - int(np.array(ycord).mean())
    return xcord,ycord

def spiral_archimedes(scan_step_pxlnb,scan_nb,obj_pxlnb):
    """" Creates the most possible successive points on spiral with step of scan_step_pxlnb,
    Returns the x,y coordinates of the spiral points.

    This is an Archimedes spiral. the equation is:
      r=(pas/2*pi)*theta
      the stepsize (radial distance between successive passes) is pas
      the curved absciss is: s(theta)=(pas/2*pi)*integral[t=0->theta](sqrt(1*t**2))dt
    """
    ylength = obj_pxlnb[0] # row
    xlength = obj_pxlnb[1] # column
    vr, vt = [0], [0]
    t = np.pi
    while len(vr) < scan_nb:
        vt.append(t)
        vr.append(scan_step_pxlnb * t / (2 * np.pi))
        t += 2 * np.pi / np.sqrt(1 + t ** 2)
    vt, vr = np.array(vt), np.array(vr)
    xcord_temp = (vr * np.cos(vt)).astype(int)
    ycord_temp = (vr * np.sin(vt)).astype(int)
    xcord = []
    ycord = []
    for idx,xcordi in enumerate(xcord_temp):
        if xcordi <= xlength and ycord_temp[idx] <= ylength:
            xcord.append(xcordi)
            ycord.append(ycord_temp[idx])
        else:
            break
    return np.array(xcord),np.array(ycord)

def make_scan(scan_type,scan_step_pxlnb,scan_nb,obj_pxlnb):
    '''
    return a tuple of (xpos,ypos)
    '''
    if scan_type == 'rect':
        posx, posy = rect(scan_step_pxlnb,scan_nb,obj_pxlnb)
    elif scan_type == 'spiral':
        posx, posy = spiral_archimedes(scan_step_pxlnb,scan_nb,obj_pxlnb)
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
    objxmid = obj_pxlnb_pad[1]//2
    objymid = obj_pxlnb_pad[0]//2

    scanxpos = scan_position[0]
    scanypos = scan_position[1]

    scanxpos = scanxpos+objxmid
    scanypos = scanypos+objymid

    left,bottom = obj_pxllim
    right = left + obj_pxlnb[1]
    top = bottom + obj_pxlnb[0]

    scanxpos_temp = []
    scanypos_temp = []
    for idx in range(len(scanxpos)):
        if right>=scanxpos[idx]>=left and top>=scanypos[idx]>=bottom:
            scanxpos_temp.append(scanxpos[idx])
            scanypos_temp.append(scanypos[idx])
        else:
            break

    if len(scanxpos_temp)<=1:
        print(f'no scan falls into the obj, generate one scan at obj center:{objxmid},{objymid}')
        scanxpos_temp.append(objxmid)
        scanypos_temp.append(objymid)      
    return scanxpos_temp,scanypos_temp

def make_dir_simu(path_dir_working):
    '''
    build a simulation folder, if the folder existed already, the folder
    will be cleared.
    '''
    current_time = time.strftime("%Y%m%d%H%M",time.localtime())
    path_dir_simulation = path_dir_working+'\\'+current_time+'_ptycho_simulation'
    path_dir_experiment = path_dir_simulation+'\\simulation_info'
    path_dir_diffraction = path_dir_simulation+'\\diffraction_patterns'
    path_dir_scanning = path_dir_experiment+'\\scanning'
    folder_list = [
                    path_dir_simulation,
                    path_dir_experiment,
                    path_dir_diffraction,
                    path_dir_scanning
                    ]
    if os.path.exists(path_dir_simulation):
        print('simulation dir exist, clearing...')
        for filename in os.listdir(path_dir_simulation):
            file_path = os.path.join(path_dir_simulation, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        for folder_path in folder_list[1:]:
            os.mkdir(folder_path)
    else:
        for folder_path in folder_list:
            os.mkdir(folder_path)
    return path_dir_simulation,path_dir_experiment,path_dir_diffraction,path_dir_scanning

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

def make_ediffraction(obj_pad,probe_Efield,scan_position_align):
    '''
    np.array is presented as (row,column),
    however the image is presented as (width,height).
    array left top is 0,0, however scan left bottom is 0,0.
    '''
    obj_rowpxlnb,obj_colunmpxlnb = obj_pad.shape
    probe_rowpxlnb,probe_colunmpxlnb = probe_Efield.shape
    scanx, scany = scan_position_align
    x0 = int(scanx-probe_colunmpxlnb//2)
    y0 = int(scany-probe_rowpxlnb//2)
    if y0<0 or x0<0 or y0+probe_rowpxlnb>obj_rowpxlnb or x0+probe_colunmpxlnb>obj_colunmpxlnb:
        print('OVER RANGE at scan position(pixels):',scanx,scany)
    if False:
        '''
        This part is used to check the product of probe and obj is correct.
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.imshow(abs(obj_pad),extent=[0,obj_colunmpxlnb,0,obj_rowpxlnb])
        ax1.scatter(scanx,scany)
        ax2 = fig.add_subplot(212)
        ax2.imshow(abs(obj_pad[-(y0+probe_rowpxlnb):-y0,x0:x0+probe_colunmpxlnb]))
        plt.pause(1)
    return probe_Efield*obj_pad[-(y0+probe_rowpxlnb):-y0,x0:x0+probe_colunmpxlnb]

def make_diffration(obj_pad,
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
    E_diffracted = make_ediffraction(obj_pad,probe_Efield,scan_position_align)
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

def save_scanning(path,fig,name):
    file = path+'\\'+str(name)+'.png'
    fig.savefig(file)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_saturation(data,cam_bitdepth):
    '''
    return the percentage of saturation
    '''
    x,y=data.shape
    pxlnb = x*y
    max_photon = 2**cam_bitdepth-1
    cam_sat_pxlnb = np.count_nonzero(data==max_photon)
    return cam_sat_pxlnb/pxlnb*100
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
obj_size = obj_pxlsize[0]*obj_pxlnb[1],obj_pxlsize[1]*obj_pxlnb[0] # width height
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
obj_size_pad = obj_pxlsize[0]*obj_pxlnb_pad[1],obj_pxlsize[1]*obj_pxlnb_pad[0] # width height
print(f'obj_pxlnb_pad:{obj_pxlnb_pad}.')
print(f'obj_size_pad:{obj_size_pad[0]:3e},{obj_size_pad[1]:3e} meter.')

# probe
'''
calculate probe information
'''
probe_shape_pxlnb = cam_pxlnb
probe_shape_size = probe_shape_pxlnb[1]*obj_pxlsize[0],probe_shape_pxlnb[0]*obj_pxlsize[1]  # width height
probe_sigma_pxlnb = probe_shape_pxlnb[1]//(2*probe_sigma_ratio[0]),probe_shape_pxlnb[0]//(2*probe_sigma_ratio[1]) # width height
probe_sigma_size = probe_sigma_pxlnb[0]*obj_pxlsize[0],probe_sigma_pxlnb[1]*obj_pxlsize[1]  # width height
print(f'probe_shape_pxlnb:{probe_shape_pxlnb}.')
print(f'probe_shape_size is:{probe_shape_size[0]:.3e},{probe_shape_size[1]:.3e} meter.')
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
scan_step_size = scan_step_pxlnb*obj_pxlsize[0]
print('scan_step_pxlnb:',scan_step_pxlnb)
print(f'scan_step_size:{scan_step_size:.3e} meter.')

if scan_step_pxlnb >= np.array(obj_pxlnb).min():
    print('obj is smaller than one scan step')
scan_position = make_scan(scan_type,scan_step_pxlnb,scan_nb,obj_pxlnb)
scan_position_align = align_scan_obj(scan_position,obj_pxlnb,obj_pxlnb_pad,obj_pxllim)

# creat dir
path_dir_simulation,path_dir_experiment,path_dir_diffraction,path_dir_scanning = make_dir_simu(path_dir_working)

# show obj
fig1 = plt.figure(tight_layout=True)

ax11 = fig1.add_subplot(121)
ax11.set_title('amplitude image')
ax11.set_xlabel('meter')
ax11.set_ylabel('meter')
ax11.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
amp_array =np.abs(obj)
ax11.imshow(amp_array,
            extent=[0,obj_pxlnb[1]*obj_pxlsize[0],0,obj_pxlnb[0]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

ax12 = fig1.add_subplot(122)
ax12.set_title('phase image')
ax12.set_xlabel('meter')
ax12.set_ylabel('meter')
ax12.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
phase_array = np.angle(obj)
ax12.imshow(phase_array,
            extent=[0,obj_pxlnb[1]*obj_pxlsize[0],0,obj_pxlnb[0]*obj_pxlsize[1]],
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
            extent=[0,probe_shape_pxlnb[1]*obj_pxlsize[0],0,probe_shape_pxlnb[0]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

ax22 = fig2.add_subplot(122)
ax22.set_title('probe log10')
ax22.set_xlabel('meter')
ax22.set_ylabel('meter')
ax22.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))
ax22.imshow(np.log10(probe_Ifield),
            extent=[0,probe_shape_pxlnb[1]*obj_pxlsize[0],0,probe_shape_pxlnb[0]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

path_probe_image = path_dir_experiment + '\\probe_image.tiff'
fig2.savefig(path_probe_image)
plt.show()

# show scan
fig3 = plt.figure(tight_layout=True)
grid = fig3.add_gridspec(6,9)

# init obj as scan position background
ax31 = fig3.add_subplot(grid[0:3,0:3])
ax31.set_title('scan position')
ax31.set_xlabel('meter')
ax31.set_ylabel('meter')
ax31.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))

ax31.imshow(np.abs(obj_pad),
            extent=[0,obj_pxlnb_pad[1]*obj_pxlsize[0],0,obj_pxlnb_pad[0]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

# init obj as scan area background
ax32 = fig3.add_subplot(grid[3:,0:3])
ax32.set_title('scan area')
ax32.set_xlabel('meter')
ax32.set_ylabel('meter')
ax32.ticklabel_format(axis='both',style='sci',scilimits=(-6,-6))

ax32.imshow(np.abs(obj_pad),
            extent=[0,obj_pxlnb_pad[1]*obj_pxlsize[0],0,obj_pxlnb_pad[0]*obj_pxlsize[1]],
            cmap='Greys_r',
            interpolation='nearest')

# init camera background as diffraction pattern background
ax33 = fig3.add_subplot(grid[:,3:])
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
                    extent=[0,obj_pxlnb_pad[1]*obj_pxlsize[0],0,obj_pxlnb_pad[0]*obj_pxlsize[1]],
                    cmap='Greys_r',
                    interpolation='nearest',
                    vmin=0,
                    vmax=2**cam_bitdepth-1)
cam_saturation = get_saturation(cam_bg,cam_bitdepth)
saturation_txt = ax33.text(0.1, 0.9, f'{cam_saturation:.3} %', transform=ax33.transAxes,
            color='r',fontweight='bold')
plt.pause(2)

# scan nb
new_scan_nb = len(scan_position_align[0])
# scan position in meter with respect to obj
scan_xposition_real = np.array(scan_position_align[0])*obj_pxlsize[0]
scan_yposition_real = np.array(scan_position_align[1])*obj_pxlsize[1]
scan_position_real = (scan_xposition_real,scan_yposition_real)
# save real scan position
path_scan_position = path_dir_experiment+'\\scan_position.csv'
with open(path_scan_position,'w+',newline='') as filecsv:
    filecsv_writer=csv.writer(filecsv,delimiter=',')
    filecsv_writer.writerows(zip(*scan_position_real)) 
# displaying colors
colors = cm.rainbow(np.linspace(0, 1, new_scan_nb))
# contour of the illumination area which were considered as non zero
circ_radius = np.array(probe_sigma_size).min()*np.array(probe_sigma_ratio).min()

cam_saturation = [] 
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
    intensity_temp = make_diffration(obj_pad,
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
    ax33.set_title('scan '+str(idx+1))
    image33.set_data(intensity_temp)

    # show saturation
    cam_saturation_i = get_saturation(intensity_temp,cam_bitdepth)
    saturation_txt.set_text(f'{cam_saturation_i:.3} %')
    cam_saturation.append(cam_saturation_i)

    # saving diffration pattern
    save_patterns(path_dir_diffraction,intensity_temp,idx+1)

    # save scanning image
    save_scanning(path_dir_scanning,fig3,idx+1)
    plt.pause(0.5)



# save saturation rate
path_cam_saturation = path_dir_experiment+'\\cam_saturation.csv'
with open(path_cam_saturation,'w+',newline='') as filecsv:
    filecsv_writer=csv.writer(filecsv,delimiter=',')
    for i in cam_saturation:
        filecsv_writer.writerow([i])

plt.pause(2)

dict_extrainfo={
    'extra_path_dirsimulation':path_dir_simulation,
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
    'probe_shape_size':probe_shape_size, # meter
    'probe_sigma_pxlnb':probe_sigma_pxlnb, # np.array((None,None))
    'probe_sigma_size':probe_sigma_size, # meter np.array((None,None))
}

dict_scaninfo={
    'scan_type_list':scan_type_list, # list of str ['rect','spiral']
    'scan_type':scan_type, # str
    'scan_recover_ratio':scan_recover_ratio, 
    'scan_nb':new_scan_nb,
}

dict_scaninfo_sup={
    'scan_step_pxlnb':scan_step_pxlnb,
    'scan_step_size':scan_step_size, # meter
}

dict_simulationinfo={
    'extra_info':dict_extrainfo,
    'obj_info':dict_objinfo,
    'obj_info_sup':dict_objinfo_sup,
    'probe_info':dict_probeinfo,
    'probe_info_sup':dict_probeinfo_sup,
    'scan_info':dict_scaninfo,
    'scan_info_sup':dict_scaninfo_sup,
    'cam_info':dict_caminfo
}

# save simulation info
path_simulation_info = path_dir_experiment+'\\simulation_info.txt'
txtfile = open(path_simulation_info,'w+') 
txt_simulationinfo = json.dumps(dict_simulationinfo,cls=NumpyEncoder)
txtfile.write(txt_simulationinfo)
txtfile.close()

# save readme
path_readme = path_dir_experiment+'\\readme.txt'
txtfile = open(path_readme,'w+') 
txtfile.write(txt_readme)
txtfile.close()