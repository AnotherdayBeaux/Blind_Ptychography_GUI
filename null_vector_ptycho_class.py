'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Jan 25 2019
@author: Zheqing Zhang
email:  zheqing@math.ucdavis.edu

implement null_vector initialization method in ptychography. Raster scan = (pertb = 0)
                                                             Pertb Raster Scan = (pertb != 0)



input_parameters = {
'savedata_path':
'n_vertical':
overlap_r'
bg_value
rank







}
'''
from blind_ptycho_fun import *
import numpy as np
import matplotlib.pyplot as plt
import os


class null_vector_ptycho(object):
    def __init__(self, input_parameters):
        # fft ifft function dict

        ptycho_fft_dict = {'ptycho_Im_PeriB_fft': ptycho_Im_PeriB_fft,
                           'ptycho_Im_FixedB_fft': ptycho_Im_FixedB_fft,
                           'ptycho_Im_PeriB_ifft': ptycho_Im_PeriB_ifft,
                           'ptycho_Im_FixedB_ifft': ptycho_Im_FixedB_ifft,
                           'pr_phase_perib_fft': pr_phase_perib_fft,
                           'pr_phase_fixedb_fft': pr_phase_fixedb_fft,
                           'pr_phase_perib_ifft': pr_phase_perib_ifft,
                           'pr_phase_fixedb_ifft': pr_phase_fixedb_ifft}

        self.savedata_path = input_parameters['savedata_path']
        self.n_v = int(input_parameters['n_vertical'])
        self.o_l = float(input_parameters['overlap_r'])
        self.bg_con = input_parameters['bg_value']

        self.rankpert = input_parameters['rank']
        if self.rankpert == 'rank one':
            self.rank_pert = 1
        else:
            self.rank_pert = 2  # full rank
        self.perturb = int(input_parameters['perturb'])

        self.image_type = input_parameters['image_type']
        if self.image_type == 'real':
            self.str1 = input_parameters['image_path']
            self.str2 = input_parameters['image_path']  # suppressed
            self.type_im = 0
        elif self.image_type == 'rand_phase':
            self.str1 = input_parameters['image_path']
            self.str2 = input_parameters['image_path']  # suppressed
            self.type_im = 1
        elif self.image_type == 'CiB_image':
            self.str1 = input_parameters['image_path_real']
            self.str2 = input_parameters['image_path_imag']
            self.type_im = 2

        self.mask_ty = input_parameters['mask_type']
        if self.mask_ty == 'Fresnel':
            self.mask_type = 2
        elif self.mask_ty == 'Correlated':
            self.mask_type = 1
        else:  # iid mask case
            self.mask_type = 0

        self.MaxIter = int(input_parameters['MaxIter'])
        self.Tol = float(input_parameters['Toler'])
        self.os_rate = int(input_parameters['os_rate'])
        self.salt_on = int(input_parameters['salt_on'])  # salton = int
        if self.salt_on == 1:
            self.salt_prob = float(input_parameters['salt_prob'])
            self.salt_inten = float(input_parameters['salt_inten'])

        # make sure the im1 and im2 are of the same size in CiB case
        self.IM1 = plt.imread(self.str1)
        self.IM2 = plt.imread(self.str2)
        if len(self.IM1.shape) > 2:  # generated a grey image
            self.IM1 = self.IM1.sum(axis=2)

        if len(self.IM2.shape) > 2:
            self.IM2 = self.IM2.sum(axis=2)

        self.Na, self.Nb = self.IM1.shape[0:2]
        self.Na1, self.Nb1 = self.IM2.shape[0:2]

        if (self.Na, self.Nb) != (self.Na1, self.Nb1):
            self.Na, self.Nb = np.min(self.Na, self.Na1), np.min(self.Nb, self.Nb1)
            print('the real part image and imag part image must have the same size')
            self.IM1 = self.IM1[:self.Na][:, :self.Nb]  # if image sizes are different, truncate them into smaller image
            self.IM2 = self.IM2[:self.Na][:, :self.Nb]

        # # for small image , test code
        # self.SUBIM = 70
        # self.Na, self.Nb = self.SUBIM, self.SUBIM
        #
        # self.IM1 = self.IM1[0:self.SUBIM][:, 0:self.SUBIM]  # test for subimage
        # self.IM1 = self.IM1.astype(np.float)
        #
        # self.IM2 = self.IM2[0:self.SUBIM][:, 0:self.SUBIM]
        # self.IM2 = self.IM2.astype(np.float)  # test for subimage
        # ##

        self.cim_diff_x, self.cim_diff_y = np.floor(self.Na / (self.n_v - 1)), np.floor(
            self.Na / (self.n_v - 1))  # cim_diff_x = cim_diff_y means square patch
        self.l_patch_x_pre, self.l_patch_y_pre = (self.cim_diff_x + 1) / (1 - self.o_l), (self.cim_diff_y + 1) / (
                    1 - self.o_l)

        self.l_patch_x = int(np.floor(self.l_patch_x_pre / 2) * 2 + 1)
        self.l_patch_y = int(np.floor(self.l_patch_y_pre / 2) * 2 + 1)
        # temp use
        self.l_patch = self.l_patch_x
        self.l_path_y = self.l_patch_x
        #

        self.beta_1 = self.l_patch_x / 2
        self.beta_2 = self.l_patch_y / 2
        self.rho = 0.5  # frensel mask parameters
        self.c_l = (self.l_patch_x, self.l_patch_y)
        self.x_line = np.arange(0, self.Na, self.cim_diff_x)
        self.x_line = self.x_line.reshape(self.x_line.shape[0], 1)

        self.y_line = np.arange(0, self.Nb, self.cim_diff_y)
        self.y_line = self.y_line.reshape(1, self.y_line.shape[0])

        self.x_c_p = self.x_line @ np.ones((1, self.y_line.shape[1]), dtype=int)
        self.y_c_p = np.ones((self.x_line.shape[0], 1), dtype=int) @ self.y_line
        self.x_c_p = self.x_c_p.astype(int)
        self.y_c_p = self.y_c_p.astype(int)

        if self.rank_pert == 1:
            self.x_c_p = self.x_c_p - (np.random.randint(self.perturb * 2 + 1, size=(self.x_c_p.shape[0], 1),
                                                         dtype='int') - self.perturb) \
                         @ np.ones((1, self.x_c_p.shape[1]), dtype='int')
            self.y_c_p = self.y_c_p - np.ones((self.y_c_p.shape[0], 1), dtype='int') \
                         @ (np.random.randint(self.perturb * 2 + 1, size=(1, self.y_c_p.shape[1]),
                                              dtype='int') - self.perturb)
        else:
            self.x_c_p = self.x_c_p - (
                        np.random.randint(self.perturb * 2 + 1, size=self.x_c_p.shape, dtype='int') - self.perturb)
            self.y_c_p = self.y_c_p - (
                        np.random.randint(self.perturb * 2 + 1, size=self.y_c_p.shape, dtype='int') - self.perturb)

        if self.bg_con == 'Fixed':
            self.fixed_bd_v = float(input_parameters['fixed_bd_value'])
            self.BackGd = self.fixed_bd_v * np.ones((3 * self.Na, 3 * self.Nb), dtype=complex)

            self.ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
            self.ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
            self.pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
            self.pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

        elif self.bg_con == 'Dark':
            self.BackGd = np.zeros((3 * self.Na, 3 * self.Nb), dtype=complex)
            self.ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
            self.ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
            self.pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
            self.pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

        else:
            self.BackGd = np.zeros((3 * self.Na, 3 * self.Nb), dtype=complex)
            self.ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_PeriB_fft']
            self.ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_PeriB_ifft']
            self.pr_phase_fft = ptycho_fft_dict['pr_phase_perib_fft']
            self.pr_phase_ifft = ptycho_fft_dict['pr_phase_perib_ifft']




        #IM, Z, phase_arg = ini_ptycho(input_parameters)

        if self.type_im == 0:
            self.IM = self.IM1
        elif self.type_im == 2:
            self.IM = self.IM1 + 1j * self.IM2
        else:
            self.IM = self.IM1 * np.exp(2j * np.pi * np.random.uniform(size=self.IM1.shape))

        if self.salt_on == 1:
            self.salt_Noise = np.random.uniform(size=self.IM.shape)
            self.salt_Noise[self.salt_Noise < (1 - self.salt_prob)] = 0
            self.salt_Noise[self.salt_Noise >= (1 - self.salt_prob)] = self.salt_inten
            self.IM = self.IM + self.salt_Noise  # need modify

        if self.mask_type == 0:  # iid mask
            self.phase_arg = np.random.uniform(size=(self.l_patch_x, self.l_patch_y))
            self.mask = np.exp(2j * np.pi * self.phase_arg)

        elif self.mask_type == 1:  # correlated mask
            self.phase_arg = Cor_Mask(self.l_patch_x,
                                 int(self.l_patch_x / 2))  # please modify the second entry half distance correlation by default
            self.mask = np.exp(2j * np.pi * self.phase_arg)
        else:  # frensel mask
            self.phase_arg = 1/2 * self.rho * ((1/self.l_patch_x * (
                    (np.arange(self.l_patch_x, dtype=float)).reshape(self.l_patch_x, 1) - self.beta_1) ** 2) @ np.ones((1, self.l_patch_y),
                                                                                                        dtype=float) + \
                                       1 / self.l_patch_y * (np.ones((self.l_patch_x, 1), dtype=float) @ (
                            np.arange(self.l_patch_y, dtype=float).reshape(1, self.l_patch_y) - self.beta_2) ** 2))
            self.mask = np.exp(2j * np.pi * self.phase_arg)

        self.Y = self.ptycho_IM_fft(self.IM, self.os_rate, self.mask, self.l_patch_x, self.x_c_p, self.y_c_p, self.BackGd)
        self.b = np.abs(self.Y)
        self.Z = self.b ** 2

        # return IM, Z, phase_arg

        self.norm_Y = np.linalg.norm(self.b, 'fro')

        # add image_xy_grid and mask_xy_grid

        self.im_x_grid = np.arange(self.Na).reshape(self.Na, 1) * np.ones((1, self.Nb), dtype='float')
        self.im_y_grid = np.ones((self.Na, 1), dtype='float') * np.arange(self.Nb).reshape(1, self.Nb)
        self.mask_x_grid = np.arange(self.l_patch_x).reshape(self.l_patch_x, 1) * np.ones((1, self.l_patch_x), dtype='float')
        self.mask_y_grid = np.ones((self.l_patch_x, 1), dtype='float') * np.arange(self.l_patch_x).reshape(1, self.l_patch_x)
        #

        # l_patch_x l_patch_y need fix

        # initial gauss on ptycho_fft(IM)
        self.lambda_t = np.random.uniform(size=self.Z.shape)

        # true mask
        self.mask = np.exp(2j*np.pi*self.phase_arg)

        # epoch counter
        self.count_DR = 1

        # residual recorder
        self.resi_DR_y = np.zeros((self.MaxIter,), dtype='float')

        # relative error image recorder (linear phase shift adjusted)
        self.relative_DR_yIMsmall1_5 = np.zeros((self.MaxIter,), dtype='float')


        # true mask
        mask = np.exp(2j * np.pi * self.phase_arg)

        # end input parameters

        self.tau = float(input_parameters['tau'])  # means we pick the tau *100 % weak signal

        self.subNa, self.subNb = self.x_c_p.shape
        self.ind_Z = np.zeros(self.Z.shape, dtype=float)
        for i in range(self.subNa):
            for j in range(self.subNb):
                self.ZK = self.Z[i * self.l_patch * self.os_rate:(i + 1) * self.l_patch * self.os_rate][:,
                     j * self.l_patch * self.os_rate:(j + 1) * self.l_patch * self.os_rate]
                self.ZK_reshape = self.ZK.reshape(-1)
                self.vec_b = self.ZK_reshape.copy()
                self.vec_b.sort()
                self.threshold = self.vec_b[int(self.tau * len(self.vec_b))]
                self.ind_z = np.ones(self.ZK.shape, dtype='float')
                self.ind_z_reshape = self.ind_z.reshape(-1)
                # if the threshold is 0, we pick the first self.tau * len(self.vec_b) zero occurrences
                if self.threshold == 0:
                    self.count = 1
                    for (self.index, self.item) in enumerate(self.ZK_reshape):
                        if self.item == 0:
                            self.ind_z_reshape[self.index] = 0.0
                            self.count += 1
                        if self.count >= self.tau * len(self.vec_b):
                            break
                    if self.vec_b[-1] <= 1e-16:
                        self.ind_z = np.zeros(self.ZK.shape, dtype=float)



                else:
                    self.ind_z[self.ZK <= self.threshold] = 0.0
                self.ind_Z[i * self.l_patch * self.os_rate:(i + 1) * self.l_patch * self.os_rate][:,
                j * self.l_patch * self.os_rate:(j + 1) * self.l_patch * self.os_rate] = self.ind_z
                # print('{}\n\n\n'.format(ind_z_reshape))
        # print(ind_Z[1:30][:,1:30])
        # len_zero = len(list(filter(lambda x: x < 0.5, ind_Z.reshape(-1))))
        # print(len_zero)
        # print(Z.shape)
        # Z1 = np.sqrt(Z)
        # norm_Y = np.linalg.norm(Z1, 'fro')


        self.IM_ = np.ones((self.Na, self.Nb), dtype='complex')
        self.p_fft_IM = self.ptycho_IM_fft(self.IM_, self.os_rate, self.mask, self.l_patch, self.x_c_p, self.y_c_p,
                                 np.ones(self.BackGd.shape, dtype=complex))
        self.nor_ptycho, self.Big_nor = self.ptycho_IM_ifft(self.p_fft_IM, self.Na, self.Nb, self.os_rate, self.mask, self.l_patch, self.x_c_p, self.y_c_p)
        self.Big_nor[self.Big_nor == 0] = 1.0


        #self.rel_error = np.zeros((self.IterA,), dtype='float') # error recorder
        self.diff_im = 1.0
        self.sqrt_nor_ptycho = np.sqrt(self.nor_ptycho)
        # find start point b
        # b_ini = np.random.uniform(size=(Na, Nb))
        # IM_ini = image initialization
        self.IM_ini = np.random.uniform(size=(self.Na, self.Nb)) * \
                      np.exp(2j*np.pi*np.random.uniform(size=(self.Na, self.Nb)))

        self.b_ini = self.IM_ini
        self.b = self.b_ini / np.linalg.norm(self.b_ini, 'fro')

        self.update_count=1


    def one_step(self):

        self.x__t = self.b/self.sqrt_nor_ptycho

        self.fft_x__t = self.ptycho_IM_fft(self.x__t, self.os_rate, self.mask, self.l_patch, self.x_c_p, self.y_c_p,
                                 np.ones(self.BackGd.shape, dtype=complex))
        self.ifft_fft_x__t, self.Big_x__t = self.ptycho_IM_ifft(self.fft_x__t*self.ind_Z, self.Na, self.Nb, self.os_rate, self.mask, self.l_patch, self.x_c_p,
                                                                self.y_c_p)

        self.b_ini = self.ifft_fft_x__t/self.sqrt_nor_ptycho
        #self.pre_b = self.b
        self.b = self.b_ini / np.linalg.norm(self.b_ini, 'fro')
        #self.diff_im = np.linalg.norm(self.pre_b - self.b, 'fro')/np.linalg.norm(self.pre_b, 'fro')
        # print('update_im = {}, diff_im = {} \n'.format(update_im, diff_im))
        self.x_t = self.b/self.sqrt_nor_ptycho * self.norm_Y

        self.nor_x_t = self.x_t / np.linalg.norm(self.x_t, 'fro') * np.linalg.norm(self.IM, 'fro')
        self.ee_im = np.abs(self.IM.reshape(-1).conj().dot(self.nor_x_t.reshape(-1))) / self.IM.reshape(-1).conj().dot(
            self.nor_x_t.reshape(-1))
        self.rel_im = np.linalg.norm(self.ee_im * self.nor_x_t - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')

        #self.rel_error[self.update_count] = self.rel_im
        print('update_im={} rel_im ={:.4e}'.format(self.update_count, self.rel_im))

        self.update_count += 1

        return (self.update_count -1), self.rel_im, self.ee_im * self.nor_x_t


