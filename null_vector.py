#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 03:41:32 2019

@author: Zheqing Zhang
"""
from blind_ptychography import *
import numpy as np

# input parameters
input_parameters = {'n_horizontal':10, 'n_vertical': 10, 'overlap_r': 0.5,
            'bg_value': 'Periodic',
            'fixed_bd_value': 20,
            'rank': 'full',
            'perturb': 3,
            'mask_type': 'IID',
            'image_type': 'CiB_image',
            'image_path_real': '/Cameraman.png',
            'image_path_imag': '/Barbara256.png',
            'mask_delta': 0.0,
            'MaxIter': 20,
            'MaxInner': 30,
            'Toler': 0.00001,
            'os_rate': 2,
            'gamma': 1,
            'salt_on': 0,
            'pois_gau':'poisson',
            'savedata_path': '/',
            'salt_noise': 0.01 }  



likelihood_dict={'poisson_likely':poisson_likely,
                 'gaussian_likely':gaussian_likely}

pois_or_gau = input_parameters['pois_gau']

if pois_or_gau == 'poisson':
    DR_update_fun = likelihood_dict['poisson_likely']

elif pois_or_gau == 'gaussian':
    DR_update_fun = likelihood_dict['gaussian_likely']

# init_ptycho #        
#input_parameters = kwargs

savedata_path = input_parameters['savedata_path'] 
#n_h = int(input_parameters['n_horizontal'])
n_v = int(input_parameters['n_vertical'])          
o_l = float(input_parameters['overlap_r'])      
bg_con = input_parameters['bg_value']

rankpert = input_parameters['rank']
if rankpert == 'rank on':
    rank_pert = 1
else:
    rank_pert = 2  # full rank
perturb = int(input_parameters['perturb'])
    

image_type = input_parameters['image_type']
if image_type == 'real':
    str1 = input_parameters['image_path']
    str2 = input_parameters['image_path']  # suppressed 
    type_im = 0
elif image_type == 'rand_phase':
    str1 = input_parameters['image_path']  
    str2 = input_parameters['image_path'] # suppressed 
    type_im = 3
elif image_type == 'CiB_image':
    str1 = input_parameters['image_path_real']
    str2 = input_parameters['image_path_imag']
    type_im = 2
    
mask_delta = float(input_parameters['mask_delta']) # mask_delta < 1/2

mask_ty = input_parameters['mask_type']
if mask_ty == 'Fresnel':
    mask_type = 2
elif mask_ty == 'Correlated':
    mask_type = 1
else:
    mask_type = 0
    

MaxIter = int(input_parameters['MaxIter'])
MaxIter_u_ip = int(input_parameters['MaxInner'])
Tol = float(input_parameters['Toler'])
gamma = float(input_parameters['gamma'])
os_rate = int(input_parameters['os_rate'])
salt_on = int(input_parameters['salt_on']) # salton = int
if salt_on == 1:
    salt_prob = float(input_parameters['salt_prob'])
    salt_inten = float(input_parameters['salt_inten'])
    


# make sure the im1 and im2 are of the same size in CiB case
IM1=plt.imread(str1)
IM2=plt.imread(str2)
Na, Nb= IM1.shape[0:2]
Na1, Nb1 = IM2.shape[0:2]

if (Na, Nb) != (Na1, Nb1):
    print('the real part image and imag part image must have the same size')

# for small image , test code
SUBIM = 70
#Na, Nb = SUBIM, SUBIM

cim_diff_x, cim_diff_y = np.floor(Na/(n_v-1)), np.floor(Na/(n_v-1))  # cim_diff_x = cim_diff_y means square patch
l_patch_x_pre, l_patch_y_pre = (cim_diff_x + 1) / (1 - o_l), (cim_diff_y + 1) / (1 - o_l)

l_patch_x = int(np.floor(l_patch_x_pre/2)*2+1)
l_patch_y = int(np.floor(l_patch_y_pre/2)*2+1)
# temp use
l_patch = l_patch_x
l_path_y = l_patch_x
#

beta_1 = l_patch_x/2
beta_2 = l_patch_y/2
rho = 0.5
c_l = (l_patch_x, l_patch_y)
x_line = np.arange(0, Na, cim_diff_x)
x_line = x_line.reshape(x_line.shape[0],1)

y_line = np.arange(0, Nb, cim_diff_y)
y_line = y_line.reshape(1, y_line.shape[0])

x_c_p = x_line @ np.ones((1,y_line.shape[1]), dtype=int)
y_c_p = np.ones((x_line.shape[0],1), dtype=int) @ y_line
x_c_p = x_c_p.astype(int)
y_c_p = y_c_p.astype(int)

if rank_pert == 1:
    x_c_p = x_c_p - (np.random.randint(perturb*2+1,size= (x_c_p.shape[0], 1), dtype = 'int')-perturb) \
            @ np.ones((1, x_c_p.shape[1]), dtype = 'int')
    y_c_p = y_c_p - np.ones((y_c_p.shape[0],1), dtype ='int') \
            @ (np.random.randint(perturb*2+1,size= (1, y_c_p.shape[1]), dtype = 'int')-perturb)
else:
    x_c_p = x_c_p - (np.random.randint(perturb*2+1,size= x_c_p.shape, dtype = 'int')-perturb)
    y_c_p = y_c_p - (np.random.randint(perturb*2+1,size= y_c_p.shape, dtype = 'int')-perturb)
    
# pass parameters to ini_ptycho(**kwargs):
pass_parameters = {'str1': str1,
                   'str2': str2,
                   'os_rate': os_rate,
                   'perturb': perturb,
                   'type_im':type_im,
                   'l_patch_x':l_patch_x,
                   'l_patch_y':l_patch_y,
                   'x_c_p': x_c_p,
                   'y_c_p': y_c_p,
                   'bd_con': bg_con,      ## str Fixed, Peridic
                   'mask_type': mask_type,
                   'c_l': c_l,            ## tuple
                   'beta_1': beta_1,
                   'beta_2': beta_2,
                   'rho': rho,
                   'salt_on': salt_on,
                   'SUBIM': SUBIM,
                   'pois_gau': input_parameters['pois_gau']
                   }

ptycho_fft_dict = {'ptycho_Im_PeriB_fft': ptycho_Im_PeriB_fft,
                   'ptycho_Im_FixedB_fft': ptycho_Im_FixedB_fft,
                   'ptycho_Im_PeriB_ifft': ptycho_Im_PeriB_ifft,
                   'ptycho_Im_FixedB_ifft': ptycho_Im_FixedB_ifft,
                   'pr_phase_perib_fft': pr_phase_perib_fft,
                   'pr_phase_fixedb_fft': pr_phase_fixedb_fft,
                   'pr_phase_perib_ifft': pr_phase_perib_ifft,
                   'pr_phase_fixedb_ifft': pr_phase_fixedb_ifft}

if pass_parameters['bd_con'] == 'Fixed':
    fixed_bd_v = float(input_parameters['fixed_bd_value'])
    BackGd = fixed_bd_v*np.ones((3*Na,3*Nb),dtype=complex)
    pass_parameters['BackGd'] = BackGd

    ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
    ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
    pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
    pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

elif pass_parameters['bd_con'] == 'Dark':
    BackGd = np.zeros((3*Na,3*Nb),dtype=complex)
    pass_parameters['BackGd'] = BackGd
    ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
    ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
    pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
    pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

else:
    BackGd = np.zeros((3 * Na, 3 * Nb), dtype=complex)
    pass_parameters['BackGd'] = BackGd
    ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_PeriB_fft']
    ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_PeriB_ifft']
    pr_phase_fft = ptycho_fft_dict['pr_phase_perib_fft']
    pr_phase_ifft = ptycho_fft_dict['pr_phase_perib_ifft']


    
if pass_parameters['salt_on'] == 1:
    pass_parameters['salt_prob'] = salt_prob
    pass_parameters['salt_inten'] = salt_inten
pass_parameters['DR_update_fun'] = DR_update_fun

IM, Z, phase_arg = ini_ptycho(pass_parameters)

Na, Nb = IM.shape
# add image_xy_grid and mask_xy_grid

# l_patch_x l_patch_y need fix 
mask_estimate = np.exp(2j*np.pi*phase_arg) * \
                np.exp(mask_delta*2j*np.pi*(np.random.uniform(size=(l_patch_x,l_patch_x))-1/2))


# true mask
mask = np.exp(2j*np.pi*phase_arg)













# end input parameters


tao = 0.5 # means we pick the 50% weak signal 

vec_b = Z.reshape(-1)
vec_b.sort()

threshold = vec_b[int(0.5*len(vec_b))]

ind_Z = np.ones(Z.shape, dtype = complex)
ind_Z[Z < threshold] = 0.0
Z[Z<threshold] = 0

Z1= np.sqrt(Z)
norm_Y = np.linalg.norm(Z1,'fro')

MaxIter_u_ip = 50
res = 1.0
Tol = 1e-6
MaxIter = 300
count = 1

while  res  > Tol and count < MaxIter: # epoch
    
    IM_=np.ones((Na, Nb), dtype='complex')
    p_fft_IM=ptycho_IM_fft(IM_, os_rate, mask_estimate, l_patch, x_c_p, y_c_p, np.ones(BackGd.shape, dtype=complex))
    nor_ptycho, Big_nor = ptycho_IM_ifft(p_fft_IM, Na, Nb, os_rate, mask_estimate, l_patch, x_c_p, y_c_p)
    Big_nor[Big_nor==0]=1.0
    residual_x=np.zeros((MaxIter_u_ip+3,), dtype='float')
    diff_im=1.0
    sqrt_nor_ptycho = np.sqrt(nor_ptycho)
    # find start point b
    b_ini = np.ones((Na, Nb), dtype = 'complex')
    b = b_ini/np.linalg.norm(b_ini,'fro')

    update_im = 1
    while (update_im < 5  or ( update_im < MaxIter_u_ip  and diff_im > 1e-2)):
        
        
        
        x__t=b/sqrt_nor_ptycho

        fft_x__t = ptycho_IM_fft(x__t, os_rate, mask_estimate, l_patch, x_c_p, y_c_p, np.ones(BackGd.shape, dtype=complex))
        ifft_fft_x__t, Big_x__t = ptycho_IM_ifft(fft_x__t*ind_Z, Na, Nb, os_rate, mask_estimate, l_patch, x_c_p, y_c_p)
        
        b_ini = ifft_fft_x__t/sqrt_nor_ptycho
        pre_b = b
        b = b_ini/np.linalg.norm(b_ini,'fro')
        diff_im = np.linalg.norm(pre_b - b,'fro')
        #print('update_im = {}, diff_im = {} \n'.format(update_im, diff_im))
        update_im +=1
        
    x_t = b/sqrt_nor_ptycho *norm_Y
    
    
    
    update_phase = 1
    
    phase_=np.ones((l_patch,l_patch),dtype='complex')
    fft_phase_=pr_phase_fft(phase_, os_rate, x_t, l_patch, x_c_p, y_c_p, Big_x__t)
    nor_phase=pr_phase_ifft(fft_phase_, os_rate, x_t, l_patch, x_c_p, y_c_p, Big_x__t)
    
    sqrt_nor_phase =np.sqrt(nor_phase)
    
    diff_phase = 1
    b_ini = np.ones((l_patch, l_patch), dtype = 'complex')
    b = b_ini/np.linalg.norm(b_ini,'fro')
    
    
    while (update_phase < 5  or ( update_phase < MaxIter_u_ip  and diff_phase > 1e-2)):
        
        
        
        phase__t=b/sqrt_nor_phase
        
        fft_phase__t = pr_phase_fft(phase__t, os_rate, x_t, l_patch, x_c_p, y_c_p, Big_x__t)
        ifft_fft_x__t = pr_phase_ifft(fft_phase__t*ind_Z, os_rate, x_t, l_patch, x_c_p, y_c_p, Big_x__t)
        
        b_ini = ifft_fft_x__t/sqrt_nor_phase
        pre_b = b
        b = b_ini/np.linalg.norm(b_ini,'fro')
        diff_phase = np.linalg.norm(pre_b - b,'fro')
        #print('update_phase = {}, diff_phase = {} \n'.format(update_phase, diff_phase))
        update_phase +=1
        
    phase_t = b/sqrt_nor_phase *norm_Y
    mask_estimate = phase_t/np.abs(phase_t)
    
    
    ee_mask = np.abs(mask.reshape(-1).conj().dot(mask_estimate.reshape(-1))) / mask.reshape(-1).conj().dot(mask_estimate.reshape(-1))
    rel_mask = np.linalg.norm(ee_mask * mask_estimate - mask, 'fro') / np.linalg.norm(mask, 'fro')
    print('count={}, rel_mask ={:.4e}'.format(count, rel_mask))
    
    count +=1 
    
        
        
        
        
    
    
    
    
    
    