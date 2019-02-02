#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 02:00:39 2018

@author: Zheqing Zhang
email:  zheqing@math.ucdavis.edu

Oversampled ptycho-fft/ifft function + any necessary function required by blind_ptycho_fun.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import spdiags


# oversampled_ifft & fft with different number of mask

# one id mask
def Nos_fft_num_mask0(X, os_rate, mask):
    Na, Nb = X.shape

    num_of_masks = 1
    Y = np.zeros((os_rate * Na, os_rate * Nb), dtype=complex)
    for kk in range(os_rate):
        LL_vec = np.exp(-2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)

        for ii in range(os_rate):
            RR_vec = np.exp(-2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)
            X1 = LL @ X @ RR
            Y1 = np.fft.fft2(X1)
            Y[kk:os_rate * Na:os_rate][:, ii:os_rate * Nb:os_rate] = Y1

    Y = Y * 1 / np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return Y


####### one random mask

def Nos_fft_num_mask1(X, os_rate, mask):
    Na, Nb = X.shape

    num_of_masks = 1
    X = mask * X
    Y = np.zeros((os_rate * Na, os_rate * Nb), dtype=complex)
    for kk in range(os_rate):
        LL_vec = np.exp(-2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)

        for ii in range(os_rate):
            RR_vec = np.exp(-2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)
            X1 = LL @ X @ RR
            Y1 = np.fft.fft2(X1)
            Y[kk:os_rate * Na:os_rate][:, ii:os_rate * Nb:os_rate] = Y1

    Y = Y * 1 / np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return Y


######## two different random mask

def Nos_fft_num_mask2(X, os_rate, mask):
    Na, Nb = X.shape

    num_of_masks = 2
    mask1X = mask[0] * X
    mask2X = mask[1] * X
    Y = np.zeros((os_rate * Na, os_rate * Nb * 2), dtype=complex)
    for kk in range(os_rate):
        LL_vec = np.exp(-2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)

        for ii in range(os_rate):
            RR_vec = np.exp(-2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)

            X1 = LL @ mask1X @ RR
            X2 = LL @ mask2X @ RR
            Y1 = np.fft.fft2(X1)
            Y2 = np.fft.fft2(X2)
            Y[kk:os_rate * Na:os_rate][:, ii:os_rate * Nb:os_rate] = Y1
            Y[kk:os_rate * Na:os_rate][:, os_rate * Nb + ii:os_rate * Nb * 2:os_rate] = Y2

    Y = Y * 1 / np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return Y


######### one id mask and one random mask

def Nos_fft_num_mask3(X, os_rate, mask):
    Na, Nb = X.shape

    num_of_masks = 3
    mask1X = mask[0] * X

    Y = np.zeros((os_rate * Na, os_rate * Nb * 2), dtype=complex)
    for kk in range(os_rate):
        LL_vec = np.exp(-2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)

        for ii in range(os_rate):
            RR_vec = np.exp(-2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)

            X1 = LL @ mask1X @ RR
            X2 = LL @ X @ RR
            Y1 = np.fft.fft2(X1)
            Y2 = np.fft.fft2(X2)
            Y[kk:os_rate * Na:os_rate][:, ii:os_rate * Nb:os_rate] = Y1
            Y[kk:os_rate * Na:os_rate][:, os_rate * Nb + ii:os_rate * Nb * 2:os_rate] = Y2

    Y = Y * 1 / np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return Y


#######   ifft part
######   one id mask
def Nos_ifft_num_mask0(Y, os_rate, mask):
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na / os_rate)
    Nb = int(Y_Nb / os_rate)
    num_of_masks = 1
    X = np.zeros((Na, Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)

            sub_Y = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:Y_Nb:os_rate])
            X1 = LL @ sub_Y @ RR
            X = X + X1

    X = X * np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return X


###### one random mask
def Nos_ifft_num_mask1(Y, os_rate, mask):
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na / os_rate)
    Nb = int(Y_Nb / os_rate)
    num_of_masks = 1
    X = np.zeros((Na, Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)

            sub_Y = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:Y_Nb:os_rate])
            X1 = LL @ sub_Y @ RR
            X = X + X1
    X = mask.conj() * X
    X = X * np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return X


###### two random mask
def Nos_ifft_num_mask2(Y, os_rate, mask):
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na / os_rate)
    Nb = int(Y_Nb / os_rate / 2)
    num_of_masks = 2
    X2 = np.zeros((Na, Nb), dtype=complex)
    X = np.zeros((Na, Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)

            sub_Y1 = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:int(Y_Nb / 2):os_rate])
            X1 = LL @ sub_Y1 @ RR
            X = X + X1

            sub_Y2 = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, int(Y_Nb / 2) + kk:Y_Nb:os_rate])
            X22 = LL @ sub_Y2 @ RR
            X2 = X2 + X22
    X = mask[0].conj() * X
    X2 = mask[1].conj() * X2
    X = X + X2
    X = X * np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return X


###### 1.5 mask
def Nos_ifft_num_mask3(Y, os_rate, mask):
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na / os_rate)
    Nb = int(Y_Nb / os_rate)
    num_of_masks = 2
    X2 = np.zeros((Na, Nb), dtype=complex)
    X = np.zeros((Na, Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2 * ii * 1j * np.pi / os_rate * np.linspace(0, (Na - 1) / Na, Na))
        LL = np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2 * kk * 1j * np.pi / os_rate * np.linspace(0, (Nb - 1) / Nb, Nb))
            RR = np.diag(RR_vec)

            sub_Y1 = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:int(Y_Nb / 2):os_rate])
            X1 = LL @ sub_Y1 @ RR
            X = X + X1

            sub_Y2 = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, int(Y_Nb / 2) + kk:Y_Nb:os_rate])
            X22 = LL @ sub_Y2 @ RR
            X2 = X2 + X22
    X = mask[0].conj() * X
    X = X + X2
    X = X * np.sqrt(Na * Nb) / np.sqrt(num_of_masks * os_rate ** 2)
    return X


###### 3 mask case
######## to be continued.
fft_dict = {'id mask': Nos_fft_num_mask0, 'one mask': Nos_fft_num_mask1, 'two mask': Nos_fft_num_mask2,
            '1.5 mask': Nos_fft_num_mask3}
ifft_dict = {'id mask': Nos_ifft_num_mask0, 'one mask': Nos_ifft_num_mask1, 'two mask': Nos_ifft_num_mask2,
             '1.5 mask': Nos_ifft_num_mask3}


#### in ptychography the num_mask is predetermined to be 1.
### image space fft/ifft
# fft periodic boundary case

def ptycho_Im_PeriB_fft(X, os_rate, mask, l_patch, x_c_p, y_c_p, BackGd):
    half_patch = int((l_patch - 1) / 2)
    Na, Nb = X.shape
    Big_X = np.kron(np.ones((3, 3), dtype=complex), X)

    subNa, subNb = x_c_p.shape
    Y = np.zeros((os_rate * l_patch * subNa, os_rate * l_patch * subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = Na + x_c_p[j][i]
            center_y = Nb + y_c_p[j][i]
            piece = Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                    center_y - half_patch:center_y + half_patch + 1]
            piece_Y = Nos_fft_num_mask1(piece, os_rate, mask)
            Y[j * os_rate * l_patch: (j + 1) * os_rate * l_patch][:,
            i * os_rate * l_patch: (i + 1) * os_rate * l_patch] = piece_Y

    return Y


# fft fixed boundary case
def ptycho_Im_FixedB_fft(X, os_rate, mask, l_patch, x_c_p, y_c_p, BackGd):
    half_patch = int((l_patch - 1) / 2)
    Na, Nb = X.shape
    Big_X = BackGd  # BackGd, if want to enforce the intensity, use BackGd = intensity
    Big_X[Na:2 * Na][:, Nb:2 * Nb] = X
    subNa, subNb = x_c_p.shape
    Y = np.zeros((os_rate * l_patch * subNa, os_rate * l_patch * subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = Na + x_c_p[j][i]
            center_y = Nb + y_c_p[j][i]
            piece = Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                    center_y - half_patch:center_y + half_patch + 1]
            piece_Y = Nos_fft_num_mask1(piece, os_rate, mask)
            Y[j * os_rate * l_patch: (j + 1) * os_rate * l_patch][:,
            i * os_rate * l_patch: (i + 1) * os_rate * l_patch] = piece_Y

    return Y


# ifft periodic boundary case
def ptycho_Im_PeriB_ifft(Y, Na, Nb, os_rate, mask, l_patch, x_c_p, y_c_p):
    half_patch = int((l_patch - 1) / 2)
    Big_X = np.zeros((3 * Na, 3 * Nb), dtype=complex)

    subNa, subNb = x_c_p.shape

    for i in range(subNa):
        for j in range(subNb):
            center_x = x_c_p[i][j] + Na
            center_y = y_c_p[i][j] + Nb
            piece_Y = Y[i * os_rate * l_patch: (i + 1) * os_rate * l_patch][:,
                      j * os_rate * l_patch: (j + 1) * os_rate * l_patch]
            piece_X = Nos_ifft_num_mask1(piece_Y, os_rate, mask)
            Big_X[center_x - half_patch:center_x + half_patch + 1][:,
            center_y - half_patch: center_y + half_patch + 1] = \
                Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                center_y - half_patch: center_y + half_patch + 1] + piece_X
    X = np.zeros((Na, Nb), dtype=complex)
    for i in range(3):
        for j in range(3):
            X = Big_X[i * Na:(i + 1) * Na][:, j * Nb:(j + 1) * Nb] + X
    return X, Big_X


# ifft fixed boundary case
def ptycho_Im_FixedB_ifft(Y, Na, Nb, os_rate, mask, l_patch, x_c_p, y_c_p):
    half_patch = int((l_patch - 1) / 2)
    Big_X = np.zeros((3 * Na, 3 * Nb), dtype=complex)

    subNa, subNb = x_c_p.shape

    for i in range(subNa):
        for j in range(subNb):
            center_x = x_c_p[i][j] + Na
            center_y = y_c_p[i][j] + Nb
            piece_Y = Y[i * os_rate * l_patch: (i + 1) * os_rate * l_patch][:,
                      j * os_rate * l_patch: (j + 1) * os_rate * l_patch]
            piece_X = Nos_ifft_num_mask1(piece_Y, os_rate, mask)
            Big_X[center_x - half_patch:center_x + half_patch + 1][:,
            center_y - half_patch: center_y + half_patch + 1] = \
                Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                center_y - half_patch: center_y + half_patch + 1] + piece_X
    X = Big_X[Na:2 * Na][:, Nb:2 * Nb]
    return X, Big_X


### mask space fft/ifft
def pr_phase_perib_fft(phase, os_rate, X, l_patch, x_c_p, y_c_p, BackGd):
    Na, Nb = X.shape
    Big_X = np.kron(np.ones((3, 3), dtype=complex),
                    X)  ### enforce the periodic boundary condition will improve performance
    ### while in fixed boundary case we don't
    half_patch = int((l_patch - 1) / 2)
    subNa, subNb = x_c_p.shape
    fft_phase = np.zeros((os_rate * l_patch * subNa, os_rate * l_patch * subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = x_c_p[j][i] + Na
            center_y = y_c_p[j][i] + Nb
            mask = Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                   center_y - half_patch: center_y + half_patch + 1]
            fft_patch_phase = Nos_fft_num_mask1(phase, os_rate, mask)
            fft_phase[os_rate * l_patch * j: os_rate * l_patch * (j + 1)][:,
            os_rate * l_patch * i: os_rate * l_patch * (i + 1)] = fft_patch_phase

    return fft_phase


####
def pr_phase_perib_ifft(fft_phase, os_rate, X, l_patch, x_c_p, y_c_p, BackGd):
    Na, Nb = X.shape
    Big_X = np.kron(np.ones((3, 3), dtype=complex), X)
    half_patch = int((l_patch - 1) / 2)
    subNa, subNb = x_c_p.shape
    phase = np.zeros((l_patch, l_patch), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = x_c_p[j][i] + Na
            center_y = y_c_p[j][i] + Nb

            mask = Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                   center_y - half_patch: center_y + half_patch + 1]
            fft_patch_phase = fft_phase[j * l_patch * os_rate:(j + 1) * l_patch * os_rate][:,
                              i * l_patch * os_rate:(i + 1) * l_patch * os_rate]
            ifft_patch_phase = Nos_ifft_num_mask1(fft_patch_phase, os_rate, mask)
            phase = phase + ifft_patch_phase

    return phase


def pr_phase_fixedb_fft(phase, os_rate, X, l_patch, x_c_p, y_c_p, BackGd):
    Na, Nb = X.shape
    Big_X = BackGd
    Big_X[Na:2 * Na][:, Nb:2 * Nb] = X
    half_patch = int((l_patch - 1) / 2)
    subNa, subNb = x_c_p.shape
    fft_phase = np.zeros((os_rate * l_patch * subNa, os_rate * l_patch * subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = x_c_p[j][i] + Na
            center_y = y_c_p[j][i] + Nb
            mask = Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                   center_y - half_patch: center_y + half_patch + 1]
            fft_patch_phase = Nos_fft_num_mask1(phase, os_rate, mask)
            fft_phase[os_rate * l_patch * j: os_rate * l_patch * (j + 1)][:,
            os_rate * l_patch * i: os_rate * l_patch * (i + 1)] = fft_patch_phase

    return fft_phase


def pr_phase_fixedb_ifft(fft_phase, os_rate, X, l_patch, x_c_p, y_c_p, BackGd):
    Na, Nb = X.shape
    Big_X = BackGd
    Big_X[Na:2 * Na][:, Nb:2 * Nb] = X
    half_patch = int((l_patch - 1) / 2)
    subNa, subNb = x_c_p.shape
    phase = np.zeros((l_patch, l_patch), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = x_c_p[j][i] + Na
            center_y = y_c_p[j][i] + Nb

            mask = Big_X[center_x - half_patch:center_x + half_patch + 1][:,
                   center_y - half_patch: center_y + half_patch + 1]
            fft_patch_phase = fft_phase[j * l_patch * os_rate:(j + 1) * l_patch * os_rate][:,
                              i * l_patch * os_rate:(i + 1) * l_patch * os_rate]
            ifft_patch_phase = Nos_ifft_num_mask1(fft_patch_phase, os_rate, mask)
            phase = phase + ifft_patch_phase

    return phase


#### poisson likelihood function Z is the  data we collected ~ fft(mask*X)**2
def poisson_likely(Q, Z, gamma):
    rot_z = (Q / gamma + np.sqrt(Q ** 2 / gamma ** 2 + 8 * (2 + 1 / gamma) * Z)) / (4 + 2 / gamma)
    return rot_z


def gaussian_likely(Q, Z, gamma):
    rot_z = (np.sqrt(Z) + 1 / gamma * Q) / (1 + 1 / gamma)
    return rot_z


# generate corelated mask's
# m = mask size; c_l= corelation distance; convolve iid phase_arg
def Cor_Mask(m, c_l):
    BigMask_phase = np.random.uniform(size=(m + c_l, m + c_l))

    BigMask = np.exp((BigMask_phase - 0.5) * np.pi * 2j)

    Cor_mask = np.zeros((m, m), dtype=complex)
    for i in range(c_l):
        for j in range(c_l):
            Cor_mask = Cor_mask + BigMask[i:m + i][:, j:m + j]

    Cor_mask = Cor_mask / np.abs(Cor_mask)
    Cor_mask = np.angle(Cor_mask) / 2 / np.pi
    return Cor_mask

# position is a list of tuples, [(p1x, p1y), (p2x, p2y), (p3x, p3y)] which picks three non zero point of mask/image
# p1x p1y -- -- -- -- - - - - p2x p2y
#  \
#  \
#  \
# p3x p3y

# n, size of mask/image
def getrid_LPS_m(f_0, f_k, n):

    Na, Nb = n[0], n[1]

    exp_2pikn = f_0 * f_k.conj()
    calib_exp = exp_2pikn * exp_2pikn[0][0].conj()
    angle_exp = np.angle(calib_exp)

    rec_k1, rec_l1 = -angle_exp[9][0] / 9 / 2 / np.pi * Na, -angle_exp[0][9] / 9 / 2 / np.pi * Nb

    return rec_k1, rec_l1


def savefigs(Iter, resi_DR_y, relative_DR_yIMsmall1_5, relative_DR_maskLPS, x__t, savefig_path):
    # x__t is already changed to real value
    plt.figure(0)
    plt.title('Error Plot')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)

    plt.semilogy(Iter[0], resi_DR_y[0], 'ro-')
    plt.semilogy(Iter[0], relative_DR_yIMsmall1_5[0], 'b^:')
    plt.semilogy(Iter[0], relative_DR_maskLPS[0], 'k*--')
    plt.legend(('Residual', 'Image Error', 'Mask Error'),
               loc='lower left')

    plt.semilogy(Iter[0:-1:3], resi_DR_y[0:-1:3], 'ro')
    plt.semilogy(Iter[0:-1:3], relative_DR_yIMsmall1_5[0:-1:3], 'b^')
    plt.semilogy(Iter[0:-1:3], relative_DR_maskLPS[0:-1:3], 'k*')

    plt.semilogy(Iter, resi_DR_y, 'r-')
    plt.semilogy(Iter, relative_DR_yIMsmall1_5, 'b:')
    plt.semilogy(Iter, relative_DR_maskLPS, 'k--')


    error_plot_location = os.path.join(savefig_path, 'error_plot.png')
    plt.savefig(error_plot_location)

    plt.close(0)
    #
    plt.figure(1)
    plt.imshow(x__t, cmap='gray')
    x__t_plot_location = os.path.join(savefig_path, 'recon_im.png')
    plt.colorbar()
    plt.grid(False)
    plt.axis('off')
    plt.savefig(x__t_plot_location)
    plt.close(0)

