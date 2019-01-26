'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Jan 25 2019
@author: Zheqing Zhang
email:  zheqing@math.ucdavis.edu

implement null_vector initialization method in ptychography. Raster scan = (pertb = 0)
                                                             Pertb Raster Scan = (pertb != 0)
'''
from blind_ptychography import *
import numpy as np
import matplotlib.pyplot as plt
import heapq


def null_vector_fun(n_vertical = 8, overlap_r = 0.5, tao_=0.5, pertb= 0, IterA=500, IM_ini=np.random.uniform(size=(256,256))):


    input_parameters = {'n_horizontal': 8, 'n_vertical': n_vertical, 'overlap_r': overlap_r,
                        'bg_value': 'Periodic',
                        'fixed_bd_value': 20,
                        'rank': 'full',
                        'perturb': pertb,
                        'mask_type': 'IID',
                        'image_type': 'rand_phase',
                        'image_path': '/Users/Beaux/Desktop/phase retrieval/image_lib/phantom.png',
                        'image_path_real': '/Users/Beaux/Desktop/phase retrieval/image_lib/phantom.png',
                        'image_path_imag': '/Barbara256.png',
                        'mask_delta': 0.1,
                        'MaxIter': 20,
                        'MaxInner': 30,
                        'Toler': 0.00001,
                        'os_rate': 2,
                        'gamma': 1,
                        'salt_on': 0,
                        'pois_gau': 'poisson',
                        'savedata_path': '/',
                        'salt_noise': 0.01}

    likelihood_dict = {'poisson_likely': poisson_likely,
                       'gaussian_likely': gaussian_likely}

    pois_or_gau = input_parameters['pois_gau']

    if pois_or_gau == 'poisson':
        DR_update_fun = likelihood_dict['poisson_likely']

    elif pois_or_gau == 'gaussian':
        DR_update_fun = likelihood_dict['gaussian_likely']

    # init_ptycho #
    # input_parameters = kwargs

    savedata_path = input_parameters['savedata_path']
    # n_h = int(input_parameters['n_horizontal'])
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
        str2 = input_parameters['image_path']  # suppressed
        type_im = 3
    elif image_type == 'CiB_image':
        str1 = input_parameters['image_path_real']
        str2 = input_parameters['image_path_imag']
        type_im = 2

    mask_delta = float(input_parameters['mask_delta'])  # mask_delta < 1/2

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
    salt_on = int(input_parameters['salt_on'])  # salton = int
    if salt_on == 1:
        salt_prob = float(input_parameters['salt_prob'])
        salt_inten = float(input_parameters['salt_inten'])

    # make sure the im1 and im2 are of the same size in CiB case
    IM1 = plt.imread(str1)
    IM2 = plt.imread(str2)
    Na, Nb = IM1.shape[0:2]
    Na1, Nb1 = IM2.shape[0:2]

    if (Na, Nb) != (Na1, Nb1):
        print('the real part image and imag part image must have the same size')

    # for small image , test code
    SUBIM = 70
    # Na, Nb = SUBIM, SUBIM

    cim_diff_x, cim_diff_y = np.floor(Na / (n_v - 1)), np.floor(
        Na / (n_v - 1))  # cim_diff_x = cim_diff_y means square patch
    l_patch_x_pre, l_patch_y_pre = (cim_diff_x + 1) / (1 - o_l), (cim_diff_y + 1) / (1 - o_l)

    l_patch_x = int(np.floor(l_patch_x_pre / 2) * 2 + 1)
    l_patch_y = int(np.floor(l_patch_y_pre / 2) * 2 + 1)
    # temp use
    l_patch = l_patch_x
    l_path_y = l_patch_x
    #

    beta_1 = l_patch_x / 2
    beta_2 = l_patch_y / 2
    rho = 0.5
    c_l = (l_patch_x, l_patch_y)
    x_line = np.arange(0, Na, cim_diff_x)
    x_line = x_line.reshape(x_line.shape[0], 1)

    y_line = np.arange(0, Nb, cim_diff_y)
    y_line = y_line.reshape(1, y_line.shape[0])

    x_c_p = x_line @ np.ones((1, y_line.shape[1]), dtype=int)
    y_c_p = np.ones((x_line.shape[0], 1), dtype=int) @ y_line
    x_c_p = x_c_p.astype(int)
    y_c_p = y_c_p.astype(int)

    if rank_pert == 1:
        x_c_p = x_c_p - (np.random.randint(perturb * 2 + 1, size=(x_c_p.shape[0], 1), dtype='int') - perturb) \
                @ np.ones((1, x_c_p.shape[1]), dtype='int')
        y_c_p = y_c_p - np.ones((y_c_p.shape[0], 1), dtype='int') \
                @ (np.random.randint(perturb * 2 + 1, size=(1, y_c_p.shape[1]), dtype='int') - perturb)
    else:
        x_c_p = x_c_p - (np.random.randint(perturb * 2 + 1, size=x_c_p.shape, dtype='int') - perturb)
        y_c_p = y_c_p - (np.random.randint(perturb * 2 + 1, size=y_c_p.shape, dtype='int') - perturb)

    # pass parameters to ini_ptycho(**kwargs):
    pass_parameters = {'str1': str1,
                       'str2': str2,
                       'os_rate': os_rate,
                       'perturb': perturb,
                       'type_im': type_im,
                       'l_patch_x': l_patch_x,
                       'l_patch_y': l_patch_y,
                       'x_c_p': x_c_p,
                       'y_c_p': y_c_p,
                       'bd_con': bg_con,  ## str Fixed, Peridic
                       'mask_type': mask_type,
                       'c_l': c_l,  ## tuple
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
        BackGd = fixed_bd_v * np.ones((3 * Na, 3 * Nb), dtype=complex)
        pass_parameters['BackGd'] = BackGd

        ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
        ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
        pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
        pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

    elif pass_parameters['bd_con'] == 'Dark':
        BackGd = np.zeros((3 * Na, 3 * Nb), dtype=complex)
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
    mask_estimate = np.exp(2j * np.pi * phase_arg) * \
                    np.exp(0.0 * 2j * np.pi * (np.random.uniform(size=(l_patch_x, l_patch_x)) - 1 / 2))

    # true mask
    mask = np.exp(2j * np.pi * phase_arg)

    # end input parameters

    tao = tao_  # means we pick the 50% weak signal

    subNa, subNb = x_c_p.shape
    ind_Z = np.zeros(Z.shape, dtype=float)
    for i in range(subNa):
        for j in range(subNb):
            ZK = Z[i*l_patch*os_rate:(i+1)*l_patch*os_rate][:,j*l_patch*os_rate:(j+1)*l_patch*os_rate]
            ZK_reshape = ZK.reshape(-1)
            vec_b = ZK_reshape.copy()
            vec_b.sort()
            threshold = vec_b[int(tao*len(vec_b))]
            ind_z = np.ones(ZK.shape,dtype='float')
            ind_z_reshape = ind_z.reshape(-1)
            if threshold == 0:
                count = 1
                for (index,item) in enumerate(ZK_reshape):
                    if item == 0:
                        ind_z_reshape[index] = 0.0
                        count += 1
                    if count >= tao * len(vec_b):
                        break
                if vec_b[-1]<= 1e-16:
                    ind_z = np.zeros(ZK.shape, dtype=float)



            else:
                ind_z[ZK <= threshold] = 0.0
            ind_Z[i*l_patch*os_rate:(i+1)*l_patch*os_rate][:,j*l_patch*os_rate:(j+1)*l_patch*os_rate] = ind_z
            #print('{}\n\n\n'.format(ind_z_reshape))
    #print(ind_Z[1:30][:,1:30])
    len_zero= len(list(filter(lambda x: x < 0.5, ind_Z.reshape(-1))))
    print(len_zero)
    print(Z.shape)
    Z1 = np.sqrt(Z)
    norm_Y = np.linalg.norm(Z1, 'fro')

    MaxIter_u_ip = 5
    res = 1.0
    Tol = 1e-6
    MaxIter = 300

    IM_ = np.ones((Na, Nb), dtype='complex')
    p_fft_IM = ptycho_IM_fft(IM_, os_rate, mask_estimate, l_patch, x_c_p, y_c_p, np.ones(BackGd.shape, dtype=complex))
    nor_ptycho, Big_nor = ptycho_IM_ifft(p_fft_IM, Na, Nb, os_rate, mask_estimate, l_patch, x_c_p, y_c_p)
    Big_nor[Big_nor == 0] = 1.0
    rel_error = np.zeros((IterA,), dtype='float')
    diff_im = 1.0
    sqrt_nor_ptycho = np.sqrt(nor_ptycho)
    # find start point b
    #b_ini = np.random.uniform(size=(Na, Nb))
    b_ini = IM_ini
    b = b_ini / np.linalg.norm(b_ini, 'fro')

    update_im = 0
    while (update_im < IterA or (update_im < MaxIter_u_ip and diff_im > 1e-2)):
        x__t = b / sqrt_nor_ptycho

        fft_x__t = ptycho_IM_fft(x__t, os_rate, mask_estimate, l_patch, x_c_p, y_c_p,
                                 np.ones(BackGd.shape, dtype=complex))
        ifft_fft_x__t, Big_x__t = ptycho_IM_ifft(fft_x__t * ind_Z, Na, Nb, os_rate, mask_estimate, l_patch, x_c_p,
                                                 y_c_p)

        b_ini = ifft_fft_x__t / sqrt_nor_ptycho
        pre_b = b
        b = b_ini / np.linalg.norm(b_ini, 'fro')
        diff_im = np.linalg.norm(pre_b - b, 'fro')
        # print('update_im = {}, diff_im = {} \n'.format(update_im, diff_im))
        x_t = b / sqrt_nor_ptycho * norm_Y

        nor_x_t = x_t/np.linalg.norm(x_t,'fro')*np.linalg.norm(IM,'fro')
        ee_im = np.abs(IM.reshape(-1).conj().dot(nor_x_t.reshape(-1))) / IM.reshape(-1).conj().dot(
            nor_x_t.reshape(-1))
        rel_im = np.linalg.norm(ee_im * nor_x_t - IM, 'fro') / np.linalg.norm(IM, 'fro')

        rel_error[update_im] = rel_im
        print('update_im={} rel_im ={:.4e}'.format(update_im, rel_im))

        update_im += 1

    #norm_x_t = np.linalg.norm(x_t,'fro')
    #print('x_t={}, norm_IM={}'.format(nor_x_t, np.linalg.norm(IM,'fro')))
    plt.imshow(np.abs(nor_x_t), cmap='gray')
    plt.title('pert={} patch={} olr={}'.format(pertb, l_patch, overlap_r))
    plt.savefig('pert={} patch={} olr={}.png'.format(pertb, l_patch, overlap_r))
    plt.show()
    plt.close()
    return rel_error, l_patch

