#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 02:00:39 2019

@author: Zheqing Zhang
email:  zheqing@math.ucdavis.edu

Blind Ptychography Class
Apply AM-DR method

"""



import numpy as np
import matplotlib.pyplot as plt
import os

from blind_ptycho_fun import *


class blind_ptycho(object):
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
        # the dict of likelihood function
        likelihood_dict = {'poisson_likely': poisson_likely,
                           'gaussian_likely': gaussian_likely}

        self.pois_or_gau = input_parameters['pois_gau']
        if self.pois_or_gau == 'poisson':
            self.DR_update_fun = likelihood_dict['poisson_likely']

        else: # pois_or_gau == 'gaussian'
            self.DR_update_fun = likelihood_dict['gaussian_likely']
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

        self.mask_delta = float(input_parameters['mask_delta'])  # mask_delta < 1/2

        self.mask_ty = input_parameters['mask_type']
        if self.mask_ty == 'Fresnel':
            self.mask_type = 2
            self.fresdevi = float(input_parameters['fresdevi'])
        elif self.mask_ty == 'Correlated':
            self.mask_type = 1
            self.cordist = float(input_parameters['cordist'])
        else:  # iid mask case
            self.mask_type = 0

        self.MaxIter = int(input_parameters['MaxIter'])
        self.MaxIter_u_ip = int(input_parameters['MaxInner'])
        self.Tol = float(input_parameters['Toler'])
        self.gamma = float(input_parameters['gamma'])
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
        # self.SUBIM = 20
        # self.Na, self.Nb = self.SUBIM, self.SUBIM
        #
        # self.IM1 = self.IM1[0:self.SUBIM][:, 0:self.SUBIM]  # test for subimage
        # self.IM1 = self.IM1.astype(np.float)
        #
        # self.IM2 = self.IM2[0:self.SUBIM][:, 0:self.SUBIM]
        # self.IM2 = self.IM2.astype(np.float)  # test for subimage
        # ##


        self.cim_diff_x, self.cim_diff_y = np.floor(self.Na/(self.n_v-1)), np.floor(
            self.Na/(self.n_v-1))  # cim_diff_x = cim_diff_y means square patch
        self.l_patch_x_pre, self.l_patch_y_pre = (self.cim_diff_x + 1) / (1 - self.o_l), (self.cim_diff_y + 1) / (1 - self.o_l)

        self.l_patch_x = int(np.floor(self.l_patch_x_pre / 2) * 2 + 1)
        self.l_patch_y = int(np.floor(self.l_patch_y_pre / 2) * 2 + 1)
        # temp use
        self.l_patch = self.l_patch_x
        self.l_path_y = self.l_patch_x
        #
        # frensel mask parameters
        self.beta_1 = self.l_patch_x / 2
        self.beta_2 = self.l_patch_y / 2

        self.x_line = np.arange(0, self.Na, self.cim_diff_x)
        self.x_line = self.x_line.reshape(self.x_line.shape[0], 1)

        self.y_line = np.arange(0, self.Nb, self.cim_diff_y)
        self.y_line = self.y_line.reshape(1, self.y_line.shape[0])

        self.x_c_p = self.x_line @ np.ones((1, self.y_line.shape[1]), dtype=int)
        self.y_c_p = np.ones((self.x_line.shape[0], 1), dtype=int) @ self.y_line
        self.x_c_p = self.x_c_p.astype(int)
        self.y_c_p = self.y_c_p.astype(int)

        if self.rank_pert == 1:
            self.x_c_p = self.x_c_p - (np.random.randint(self.perturb * 2 + 1, size=(self.x_c_p.shape[0], 1), dtype='int') - self.perturb) \
                    @ np.ones((1, self.x_c_p.shape[1]), dtype='int')
            self.y_c_p = self.y_c_p - np.ones((self.y_c_p.shape[0], 1), dtype='int') \
                    @ (np.random.randint(self.perturb * 2 + 1, size=(1, self.y_c_p.shape[1]), dtype='int') - self.perturb)
        else:
            self.x_c_p = self.x_c_p - (np.random.randint(self.perturb * 2 + 1, size=self.x_c_p.shape, dtype='int') - self.perturb)
            self.y_c_p = self.y_c_p - (np.random.randint(self.perturb * 2 + 1, size=self.y_c_p.shape, dtype='int') - self.perturb)



        if self.bg_con == 'Fixed':
            self.fixed_bd_v = float(input_parameters['fixed_bd_value'])
            self.BackGd = self.fixed_bd_v * np.ones((3*self.Na, 3*self.Nb), dtype=complex)

            self.ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
            self.ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
            self.pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
            self.pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

        elif self.bg_con == 'Dark':
            self.BackGd = np.zeros((3*self.Na, 3*self.Nb), dtype=complex)
            self.ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
            self.ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
            self.pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
            self.pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

        else:
            self.BackGd = np.zeros((3*self.Na, 3*self.Nb), dtype=complex)
            self.ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_PeriB_fft']
            self.ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_PeriB_ifft']
            self.pr_phase_fft = ptycho_fft_dict['pr_phase_perib_fft']
            self.pr_phase_ifft = ptycho_fft_dict['pr_phase_perib_ifft']


        # start to initialize blind-ptycho by generating diffraction pattern data
        # and image and mask

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
                                 int(self.l_patch_x * self.cordist))  # please modify the second entry half distance correlation by default
            self.mask = np.exp(2j * np.pi * self.phase_arg)
        else:  # frensel mask
            self.phase_arg = 1/2 / self.fresdevi * ((1/self.l_patch_x * (
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






        # l_patch_x l_patch_y need fix
        self.mask_estimate = np.exp(2j * np.pi * self.phase_arg) * \
                        np.exp(self.mask_delta * 2j * np.pi * (np.random.uniform(size=(self.l_patch_x, self.l_patch_x)) - 1 / 2))

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

        # relative error mask recorder (linear phase shift adjusted)
        self.relative_DR_maskLPS = np.zeros((self.MaxIter,), dtype='float')


    # end __init__ class function







    def one_step(self):

        self.update_im = 1

        # calculate nor_factors depenent on mask_estimate
        self.IM_ = np.ones((self.Na, self.Nb), dtype='complex')
        self.p_fft_IM = self.ptycho_IM_fft(self.IM_, self.os_rate, self.mask_estimate, self.l_patch, self.x_c_p,
                                           self.y_c_p,
                                           np.ones(self.BackGd.shape, dtype=complex))
        self.nor_ptycho, self.Big_nor = self.ptycho_IM_ifft(self.p_fft_IM, self.Na, self.Nb, self.os_rate,
                                                            self.mask_estimate, self.l_patch, self.x_c_p, self.y_c_p)
        self.Big_nor[self.Big_nor == 0] = 1.0
        self.residual_x = np.zeros((self.MaxIter_u_ip + 3,), dtype='float')
        self.resi_diff_im = 1.0

        while (self.update_im < 5 or (self.update_im < self.MaxIter_u_ip and self.resi_diff_im > 1e-5)):
            self.x_t, self.Big_x_t = self.ptycho_IM_ifft(self.lambda_t, self.Na, self.Nb, self.os_rate,
                                                         self.mask_estimate, self.l_patch, self.x_c_p, self.y_c_p)
            self.x__t = self.x_t / self.nor_ptycho
            self.Big_x__t = self.Big_x_t / self.Big_nor
            self.AAlambda = self.ptycho_IM_fft(self.x__t, self.os_rate, self.mask_estimate, self.l_patch, self.x_c_p,
                                               self.y_c_p, self.Big_x__t)
            # AAlambda=ptycho_Im_PeriB_fft(x__t,os_rate,mask_estimate,l_patch,x_c_p,y_c_p,Big_x__t)
            # Big_x__t: no enforce of prior info
            self.y_tplus1 = self.AAlambda

            self.ylambda_k = 2 * self.y_tplus1 - self.lambda_t
            self.Q = np.abs(self.ylambda_k)

            self.rot_z = self.DR_update_fun(self.Q, self.Z, self.gamma)

            self.z_tplus1 = self.rot_z * self.ylambda_k / self.Q
            self.lambda_t = self.lambda_t + self.z_tplus1 - self.y_tplus1

            # calculate the termination error
            # ee=np.abs(IM.reshape(-1).conj().dot(x__t.reshape(-1)))/IM.reshape(-1).conj().dot(x__t.reshape(-1))
            # rel_xx = np.linalg.norm(ee*x__t-IM, 'fro')/np.linalg.norm(IM,'fro')

            self.res_x = np.linalg.norm(np.abs(self.b) - np.abs(self.y_tplus1), 'fro') / self.norm_Y

            self.residual_x[self.update_im + 2] = self.res_x
            self.resi_diff_im = 1 / 3 * np.linalg.norm(
                self.residual_x[self.update_im:self.update_im + 3] - self.residual_x[
                                                                     self.update_im - 1:self.update_im + 2],
                1) / self.res_x
            self.update_im = self.update_im + 1




        self.update_phase = 1

        # # calculate nor_factors depenent on image
        self.phase_ = np.ones((self.l_patch, self.l_patch), dtype='complex')
        self.fft_phase_ = self.pr_phase_fft(self.phase_, self.os_rate, self.x__t, self.l_patch, self.x_c_p,
                                            self.y_c_p, self.Big_x__t)
        self.nor_phase = self.pr_phase_ifft(self.fft_phase_, self.os_rate, self.x__t, self.l_patch, self.x_c_p,
                                            self.y_c_p, self.Big_x__t)

        self.fft_phase = self.y_tplus1  ##
        self.residual_xx = np.zeros((self.MaxIter_u_ip + 3,), dtype='float')
        self.resi_diff_phase = 1.0

        while self.update_phase < 5 or (self.update_phase < self.MaxIter_u_ip and self.resi_diff_phase > 1e-5):
            self.nor_mask_tplus1 = self.pr_phase_ifft(self.fft_phase, self.os_rate, self.x__t, self.l_patch,
                                                      self.x_c_p, self.y_c_p, self.Big_x__t) / self.nor_phase

            self.mask_tplus1 = self.nor_mask_tplus1 / np.abs(self.nor_mask_tplus1)  # enforce unimodular constraint
            self.fft_phase_1 = self.pr_phase_fft(self.mask_tplus1, self.os_rate, self.x__t, self.l_patch,
                                                 self.x_c_p, self.y_c_p, self.Big_x__t)
            self.Q_phase = 2 * self.fft_phase_1 - self.fft_phase
            self.Q = np.abs(self.Q_phase)
            self.rot_z = self.DR_update_fun(self.Q, self.Z, self.gamma)
            self.z_mask_tplus1 = self.rot_z * self.Q_phase / self.Q
            self.fft_phase = self.fft_phase + self.z_mask_tplus1 - self.fft_phase_1

            self.res_mask = np.linalg.norm(np.abs(self.fft_phase_1) - self.b, 'fro') / self.norm_Y  # b =sqrt(Z)
            self.residual_xx[self.update_phase + 2] = self.res_mask
            self.resi_diff_phase = 1 / 3 * np.linalg.norm(
                self.residual_xx[self.update_phase:self.update_phase + 3] - self.residual_xx[
                                                                            self.update_phase - 1:self.update_phase + 2],
                1) / self.res_mask
            self.update_phase = self.update_phase + 1
            # end mask update






        # calculate the termination error


        self.mask_estimate = self.mask_tplus1
        self.lambda_t = self.fft_phase_1

        # calculate mask LPS error
        self.ee_mask = np.abs(self.mask.reshape(-1).conj().dot(self.mask_estimate.reshape(-1))) / self.mask.reshape(
            -1).conj().dot(
            self.mask_estimate.reshape(-1))
        self.recm_k1, self.recm_l1 = getrid_LPS_m(self.mask, self.ee_mask * self.mask_estimate, self.IM.shape)
        self.mask_t_LPS = np.exp(-2j * np.pi * (
                    self.mask_x_grid * self.recm_k1 / self.Na + self.mask_y_grid * self.recm_l1 / self.Nb)) * self.mask_estimate
        self.ee_mask = np.abs(self.mask.reshape(-1).conj().dot(self.mask_t_LPS.reshape(-1))) / self.mask.reshape(
            -1).conj().dot(
            self.mask_t_LPS.reshape(-1))
        self.rel_LPS_mask = np.linalg.norm(self.ee_mask * self.mask_t_LPS - self.mask, 'fro') / np.linalg.norm(
            self.mask, 'fro')

        # calculate rel_LPS_x
        self.ee_p = np.abs(self.IM.reshape(-1).conj().dot(self.x__t.reshape(-1))) / self.IM.reshape(-1).conj().dot(
            self.x__t.reshape(-1))
        self.rec_k1, self.rec_l1 = getrid_LPS_m(self.IM, self.ee_p * self.x__t,
                                                self.IM.shape)  # n subim dimension. ###
        # rec_k1, rec_l1 = -recm_k1, -recm_l1
        self.LPS_x_t = np.exp(-2j * np.pi * (
                    self.im_x_grid * self.rec_k1 / self.Na + self.im_y_grid * self.rec_l1 / self.Nb)) * self.x__t
        self.ee = np.abs(self.IM.reshape(-1).conj().dot(self.LPS_x_t.reshape(-1))) / (
            self.IM.reshape(-1).conj().dot(self.LPS_x_t.reshape(-1)))
        self.rel_LPS_x = np.linalg.norm(self.LPS_x_t * self.ee - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')
        # rel_x=np.linalg.norm(ee_p*x__t-IM,'fro')/np.linalg.norm(IM,'fro')

        # store data
        self.resi_DR_y[self.count_DR - 1] = self.res_x
        self.relative_DR_yIMsmall1_5[self.count_DR - 1] = self.rel_LPS_x
        # relative_DR_xIMsmall1_5[count_DR-1] =rel_x
        self.relative_DR_maskLPS[self.count_DR - 1] = self.rel_LPS_mask

        print('count_DR=%d rec_k1=%s rec_l1=%s \n rel_LPS_x=%s rel_LPS_mask=%s res_x=%s \n' % (
            self.count_DR, '{:.4e}'.format(self.rec_k1), '{:.4e}'.format(self.rec_l1),
            '{:.4e}'.format(self.rel_LPS_x),
            '{:.4e}'.format(self.rel_LPS_mask),
            '{:.4e}'.format(self.res_x)))

        self.count_DR += 1

        return self.res_x, self.rel_LPS_x, self.rel_LPS_mask, self.count_DR, self.LPS_x_t * self.ee

    ''' under construction'''
    # def image_update(self, mask_estimate, lambda_t):
    #     self.lambda_t1 = lambda_t
    #     self.mask_estimate1 = mask_estimate
    #     self.update_im = 1
    #
    #     # calculate nor_factors depenent on mask_estimate
    #     self.IM_ = np.ones((self.Na, self.Nb), dtype='complex')
    #     self.p_fft_IM = self.ptycho_IM_fft(self.IM_, self.os_rate, self.mask_estimate1, self.l_patch, self.x_c_p, self.y_c_p,
    #                              np.ones(self.BackGd.shape, dtype=complex))
    #     self.nor_ptycho, self.Big_nor = self.ptycho_IM_ifft(self.p_fft_IM, self.Na, self.Nb, self.os_rate, self.mask_estimate1, self.l_patch, self.x_c_p, self.y_c_p)
    #     self.Big_nor[self.Big_nor == 0] = 1.0
    #     self.residual_x = np.zeros((self.MaxIter_u_ip + 3,), dtype='float')
    #     self.resi_diff_im = 1.0
    #
    #     while (self.update_im < 5 or (self.update_im < self.MaxIter_u_ip and self.resi_diff_im > 1e-5)):
    #         self.x_t, self.Big_x_t = self.ptycho_IM_ifft(self.lambda_t1, self.Na, self.Nb, self.os_rate, self.mask_estimate1, self.l_patch, self.x_c_p, self.y_c_p)
    #         self.x__t = self.x_t / self.nor_ptycho
    #         self.Big_x__t = self.Big_x_t / self.Big_nor
    #         self.AAlambda = self.ptycho_IM_fft(self.x__t, self.os_rate, self.mask_estimate1, self.l_patch, self.x_c_p, self.y_c_p, self.Big_x__t)
    #         # AAlambda=ptycho_Im_PeriB_fft(x__t,os_rate,mask_estimate,l_patch,x_c_p,y_c_p,Big_x__t)
    #         # Big_x__t: no enforce of prior info
    #         self.y_tplus1 = self.AAlambda
    #
    #         self.ylambda_k = 2 * self.y_tplus1 - self.lambda_t1
    #         self.Q = np.abs(self.ylambda_k)
    #
    #         self.rot_z = self.DR_update_fun(self.Q, self.Z, self.gamma)
    #
    #         self.z_tplus1 = self.rot_z*self.ylambda_k/self.Q
    #         self.lambda_t1 = self.lambda_t1 + self.z_tplus1 - self.y_tplus1
    #
    #         # calculate the termination error
    #         # ee=np.abs(IM.reshape(-1).conj().dot(x__t.reshape(-1)))/IM.reshape(-1).conj().dot(x__t.reshape(-1))
    #         # rel_xx = np.linalg.norm(ee*x__t-IM, 'fro')/np.linalg.norm(IM,'fro')
    #
    #         self.res_x = np.linalg.norm(np.abs(self.b) - np.abs(self.y_tplus1), 'fro') / self.norm_Y
    #
    #         self.residual_x[self.update_im + 2] = self.res_x
    #         self.resi_diff_im = 1 / 3 * np.linalg.norm(
    #             self.residual_x[self.update_im:self.update_im + 3] -self.residual_x[self.update_im-1:self.update_im + 2], 1)/self.res_x
    #         self.update_im = self.update_im + 1
    #
    #         return self.x__t, self.Big_x__t, self.y_tplus1
    #
    #
    # def mask_update(self, x__t, Big_x__t, y_tplus1):
    #     self.x__t = x__t
    #     self.Big_x__t = Big_x__t
    #     self.y_tplus1 = y_tplus1
    #
    #     self.update_phase = 1
    #
    #     # # calculate nor_factors depenent on image
    #     self.phase_ = np.ones((self.l_patch, self.l_patch), dtype='complex')
    #     self.fft_phase_ = self.pr_phase_fft(self.phase_, self.os_rate, self.x__t, self.l_patch, self.x_c_p, self.y_c_p, self.Big_x__t)
    #     self.nor_phase = self.pr_phase_ifft(self.fft_phase_, self.os_rate, self.x__t, self.l_patch, self.x_c_p, self.y_c_p, self.Big_x__t)
    #
    #     self.fft_phase = self.y_tplus1  ##
    #     self.residual_xx = np.zeros((self.MaxIter_u_ip + 3,), dtype='float')
    #     self.resi_diff_phase = 1.0
    #
    #     while  self.update_phase < 5 or (self.update_phase < self.MaxIter_u_ip and self.resi_diff_phase > 1e-5):
    #         self.nor_mask_tplus1 = self.pr_phase_ifft(self.fft_phase, self.os_rate, self.x__t, self.l_patch, self.x_c_p, self.y_c_p, self.Big_x__t)/self.nor_phase
    #
    #         self.mask_tplus1 = self.nor_mask_tplus1 / np.abs(self.nor_mask_tplus1)  # enforce unimodular constraint
    #         self.fft_phase_1 = self.pr_phase_fft(self.mask_tplus1, self.os_rate, self.x__t, self.l_patch, self.x_c_p, self.y_c_p, self.Big_x__t)
    #         self.Q_phase = 2*self.fft_phase_1-self.fft_phase
    #         self.Q = np.abs(self.Q_phase)
    #         self.rot_z = self.DR_update_fun(self.Q, self.Z, self.gamma)
    #         self.z_mask_tplus1 =self.rot_z * self.Q_phase / self.Q
    #         self.fft_phase = self.fft_phase + self.z_mask_tplus1 - self.fft_phase_1
    #
    #         # calculate the termination error
    #         self.res_mask = np.linalg.norm(np.abs(self.fft_phase_1) - self.b, 'fro')/self.norm_Y  # b =sqrt(Z)
    #         self.residual_xx[self.update_phase + 2] = self.res_mask
    #         self.resi_diff_phase = 1 / 3 * np.linalg.norm(
    #             self.residual_xx[self.update_phase:self.update_phase + 3] - self.residual_xx[self.update_phase - 1:self.update_phase + 2],
    #             1)/self.res_mask
    #         self.update_phase = self.update_phase + 1
    #
    #     return self.mask_tplus1, self.fft_phase_1
    #
    # def one_AM_step(self):
    #     #self.mask_estimate, self.lambda_t = mask_estimate, lambda_t
    #
    #     self.x__t, self.Big_x__t, self.y_tplus1 = self.image_update(self.mask_estimate, self.lambda_t)
    #     self.mask_estimate, self.lambda_t = self.mask_update(self.x__t, self.Big_x__t, self.y_tplus1)
    #
    #     # calculate error
    #     self.ee_mask = np.abs(self.mask.reshape(-1).conj().dot(self.mask_estimate.reshape(-1))) / self.mask.reshape(-1).conj().dot(
    #         self.mask_estimate.reshape(-1))
    #     self.recm_k1, self.recm_l1 = getrid_LPS_m(self.mask, self.ee_mask*self.mask_estimate, self.IM.shape)
    #     self.mask_t_LPS = np.exp(-2j*np.pi * (self.mask_x_grid*self.recm_k1/self.Na+self.mask_y_grid*self.recm_l1/self.Nb))*self.mask_estimate
    #     self.ee_mask = np.abs(self.mask.reshape(-1).conj().dot(self.mask_t_LPS.reshape(-1)))/self.mask.reshape(-1).conj().dot(
    #         self.mask_t_LPS.reshape(-1))
    #     self.rel_LPS_mask = np.linalg.norm(self.ee_mask*self.mask_t_LPS-self.mask, 'fro')/np.linalg.norm(self.mask, 'fro')
    #
    #     # calculate rel_LPS_x
    #     self.ee_p = np.abs(self.IM.reshape(-1).conj().dot(self.x__t.reshape(-1))) / self.IM.reshape(-1).conj().dot(self.x__t.reshape(-1))
    #     self.rec_k1, self.rec_l1 = getrid_LPS_m(self.IM, self.ee_p * self.x__t, self.IM.shape)  # n subim dimension. ###
    #     # rec_k1, rec_l1 = -recm_k1, -recm_l1
    #     self.LPS_x_t = np.exp(-2j * np.pi * (self.im_x_grid * self.rec_k1 / self.Na + self.im_y_grid * self.rec_l1 / self.Nb)) * self.x__t
    #     self.ee = np.abs(self.IM.reshape(-1).conj().dot(self.LPS_x_t.reshape(-1))) / (self.IM.reshape(-1).conj().dot(self.LPS_x_t.reshape(-1)))
    #     self.rel_LPS_x = np.linalg.norm(self.LPS_x_t * self.ee - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')
    #     # rel_x=np.linalg.norm(ee_p*x__t-IM,'fro')/np.linalg.norm(IM,'fro')
    #
    #     # store data
    #     self.resi_DR_y[self.count_DR - 1] = self.res_x
    #     self.relative_DR_yIMsmall1_5[self.count_DR - 1] = self.rel_LPS_x
    #     # relative_DR_xIMsmall1_5[count_DR-1] =rel_x
    #     self.relative_DR_maskLPS[self.count_DR - 1] = self.rel_LPS_mask
    #
    #     self.count_DR += 1
    #
    #     print('count_DR=%d rec_k1=%s rec_l1=%s \n rel_LPS_x=%s rel_LPS_mask=%s res_x=%s \n' % (
    #         self.count_DR, '{:.4e}'.format(self.rec_k1), '{:.4e}'.format(self.rec_l1), '{:.4e}'.format(self.rel_LPS_x),
    #         '{:.4e}'.format(self.rel_LPS_mask),
    #         '{:.4e}'.format(self.res_x)))
    #
    #     return self.res_x, self.count_DR



