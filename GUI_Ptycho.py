#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 19 12:55:15 2018

PYTHON TKINTER GUI For phase retreival ptychography
@author: Zheqing Zhang 
Email: zheqing@math.ucdavis.edu
"""
# import matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from blind_ptycho_fun import *

from blind_ptycho_class import *
from null_vector_ptycho_class import *


# import threading allowing animate running seperately.
import threading

# import numpy
import numpy as np

# import tkinter
import tkinter as tk
import tkinter.messagebox as msb
from tkinter import ttk
from tkinter import filedialog
# add scrobar-text widget
import tkinter.scrolledtext as tkst

import datetime
import os


def gettipsnv():
    msb.showinfo('n_veritial',
                 'Number of small patches you want to split the image into vertically.'
                 'Positive integer. Once fixed, the square patch size is determined')


def gettipcordist():
    msb.showinfo('Corr dist',
                 'Cor Dist regulates the extend that pixels in mask are correlated'
                 'It is a number in [0, 1]. 0: uncorrelated mask == iid mask'
                 '1: maximum correlation. Pixels are correlated to each other.')
def gettipfresdevi():
    msb.showinfo('Fres Devi',
                 'Standard deviation we use to generate fresnel mask')





def gettipsnh():
    msb.showinfo('n_horizontal','Number of small patches you want to split the image into horizontally.')


def gettipsolr():
    msb.showinfo('OLR','Overlaping ratio between two adjacent measurements')

# for blind_ptychography
def bg_selection(*args):
    
    if bg_value.get() == 'Fixed':
        fixed_bd_label.config(state='normal')
        fixed_bd_value.config(state='normal',relief='solid')
        
    if bg_value.get() != 'Fixed':
        fixed_bd_value.delete(0,'end')
        fixed_bd_label.config(state='disabled')
        fixed_bd_value.config(state='disabled', relief='sunken')


# for null vector
def bg_selection_null_v(*args):
    if bg_value_null_v.get() == 'Fixed':
        fixed_bd_label_null_v.config(state='normal')
        fixed_bd_value_null_v.config(state='normal', relief='solid')

    if bg_value_null_v.get() != 'Fixed':
        fixed_bd_value_null_v.delete(0, 'end')
        fixed_bd_label_null_v.config(state='disabled')
        fixed_bd_value_null_v.config(state='disabled', relief='sunken')

# for blind ptycho
def getrank():
    rank = rank_per.get()
    return rank

# for null vector
def getrank_null_v():
    rank = rank_per_null_v.get()
    return rank

# for blind ptycho
def getimage_t():
    image_type = image_ty.get()
    if image_type == 'real':
        image_path_real_entry.config(state='normal', relief = 'solid')
        image_path_real_button.config(state = 'normal')
        image_path_rand_entry.config(state='disabled', relief = 'sunken')
        image_path_rand_button.config(state = 'disabled')
        image_path_rand.set(' ')
        image_path_CiB_real_entry.config(state='disabled', relief = 'sunken')
        image_path_CiB_real_button.config(state = 'disabled')
        image_path_CiB_real.set(' ')
        image_path_CiB_imag_entry.config(state='disabled', relief = 'sunken')
        image_path_CiB_imag_button.config(state = 'disabled')
        image_path_CiB_imag.set(' ')
        
    if image_type == 'rand_phase':
        image_path_real_entry.config(state='disabled', relief = 'sunken')
        image_path_real_button.config(state = 'disabled')
        image_path_real.set(' ')
        image_path_rand_entry.config(state='normal', relief = 'solid')
        image_path_rand_button.config(state = 'normal')
        image_path_CiB_real_entry.config(state='disabled', relief = 'sunken')
        image_path_CiB_real_button.config(state = 'disabled')
        image_path_CiB_real.set(' ')
        image_path_CiB_imag_entry.config(state='disabled', relief = 'sunken')
        image_path_CiB_imag_button.config(state = 'disabled')
        image_path_CiB_imag.set(' ')
    
    if image_type == 'CiB_image':
        image_path_real_entry.config(state='disabled', relief = 'sunken')
        image_path_real_button.config(state = 'disabled')
        image_path_real.set(' ')
        image_path_rand_entry.config(state='disabled', relief = 'sunken')
        image_path_rand_button.config(state = 'disabled')
        image_path_rand.set(' ')
        image_path_CiB_real_entry.config(state='normal', relief = 'solid')
        image_path_CiB_real_button.config(state = 'normal')
        image_path_CiB_imag_entry.config(state='normal', relief = 'solid')
        image_path_CiB_imag_button.config(state = 'normal')
    return image_type


def selectPath_real():  
    path_ = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('png files','*.png'),('jpeg','*.jpg')))
    image_path_real.set(path_) 


def selectPath_rand():
    path_ = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('png files','*.png'),('jpeg','*.jpg')))
    image_path_rand.set(path_) 


def selectPath_CiB_real():
    path_ = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('png files','*.png'),('jpeg','*.jpg')))
    image_path_CiB_real.set(path_) 


def selectPath_CiB_imag():
    path_ = filedialog.askopenfilename(initialdir = '/',title = 'Select file',filetypes = (('png files','*.png'),('jpeg','*.jpg')))
    image_path_CiB_imag.set(path_)


# for null vector
def getimage_t_null_v():
    image_type = image_ty_null_v.get()
    if image_type == 'real':
        image_path_real_entry_null_v.config(state='normal', relief='solid')
        image_path_real_button_null_v.config(state='normal')
        image_path_rand_entry_null_v.config(state='disabled', relief='sunken')
        image_path_rand_button_null_v.config(state='disabled')
        image_path_rand_null_v.set(' ')
        image_path_CiB_real_entry_null_v.config(state='disabled', relief='sunken')
        image_path_CiB_real_button_null_v.config(state='disabled')
        image_path_CiB_real_null_v.set(' ')
        image_path_CiB_imag_entry_null_v.config(state='disabled', relief='sunken')
        image_path_CiB_imag_button_null_v.config(state='disabled')
        image_path_CiB_imag_null_v.set(' ')

    if image_type == 'rand_phase':
        image_path_real_entry_null_v.config(state='disabled', relief='sunken')
        image_path_real_button_null_v.config(state='disabled')
        image_path_real_null_v.set(' ')
        image_path_rand_entry_null_v.config(state='normal', relief='solid')
        image_path_rand_button_null_v.config(state='normal')
        image_path_CiB_real_entry_null_v.config(state='disabled', relief='sunken')
        image_path_CiB_real_button_null_v.config(state='disabled')
        image_path_CiB_real_null_v.set(' ')
        image_path_CiB_imag_entry_null_v.config(state='disabled', relief='sunken')
        image_path_CiB_imag_button_null_v.config(state='disabled')
        image_path_CiB_imag_null_v.set(' ')

    if image_type == 'CiB_image':
        image_path_real_entry_null_v.config(state='disabled', relief='sunken')
        image_path_real_button_null_v.config(state='disabled')
        image_path_real_null_v.set(' ')
        image_path_rand_entry_null_v.config(state='disabled', relief='sunken')
        image_path_rand_button_null_v.config(state='disabled')
        image_path_rand_null_v.set(' ')
        image_path_CiB_real_entry_null_v.config(state='normal', relief='solid')
        image_path_CiB_real_button_null_v.config(state='normal')
        image_path_CiB_imag_entry_null_v.config(state='normal', relief='solid')
        image_path_CiB_imag_button_null_v.config(state='normal')
    return image_type


# for blind ptychography
def enable_mask_blind_ptycho(*args):
    if mask_type_value.get() == 'Correlated':
        cordis_label.config(state='normal')
        cordis_entry.config(state='normal', relief='solid')

        fresnel_devi.set('')
        fresnel_label.config(state='disabled')
        fresnel_devi_entry.config(state='disabled', relief='sunken')

    elif mask_type_value.get() == 'Fresnel':
        fresnel_label.config(state='normal')
        fresnel_devi_entry.config(state='normal', relief='solid')

        cordis.set('')
        cordis_label.config(state='disabled')
        cordis_entry.config(state='disabled', relief='sunken')

    else:
        cordis.set('')
        cordis_label.config(state='disabled')
        cordis_entry.config(state='disabled', relief='sunken')

        fresnel_devi.set('')
        fresnel_label.config(state='disabled')
        fresnel_devi_entry.config(state='disabled', relief='sunken')



# for null vector
def enable_mask_null_v(*args):
    if mask_type_value_null_v.get() == 'Correlated':
        cordis_label_null_v.config(state='normal')
        cordis_entry_null_v.config(state='normal', relief='solid')

        fresnel_devi_null_v.set('')
        fresnel_label_null_v.config(state='disabled')
        fresnel_devi_entry_null_v.config(state='disabled', relief='sunken')

    elif mask_type_value_null_v.get() == 'Fresnel':
        fresnel_label_null_v.config(state='normal')
        fresnel_devi_entry_null_v.config(state='normal', relief='solid')

        cordis_null_v.set('')
        cordis_label_null_v.config(state='disabled')
        cordis_entry_null_v.config(state='disabled', relief='sunken')

    else:
        cordis_null_v.set('')
        cordis_label_null_v.config(state='disabled')
        cordis_entry_null_v.config(state='disabled', relief='sunken')

        fresnel_devi_null_v.set('')
        fresnel_label_null_v.config(state='disabled')
        fresnel_devi_entry_null_v.config(state='disabled', relief='sunken')




def selectPath_real_null_v():
    path_ = filedialog.askopenfilename(initialdir='/', title='Select file',
                                       filetypes=(('png files', '*.png'), ('jpeg', '*.jpg')))
    image_path_real_null_v.set(path_)


def selectPath_rand_null_v():
    path_ = filedialog.askopenfilename(initialdir='/', title='Select file',
                                       filetypes=(('png files', '*.png'), ('jpeg', '*.jpg')))
    image_path_rand_null_v.set(path_)


def selectPath_CiB_real_null_v():
    path_ = filedialog.askopenfilename(initialdir='/', title='Select file',
                                       filetypes=(('png files', '*.png'), ('jpeg', '*.jpg')))
    image_path_CiB_real_null_v.set(path_)


def selectPath_CiB_imag_null_v():
    path_ = filedialog.askopenfilename(initialdir='/', title='Select file',
                                       filetypes=(('png files', '*.png'), ('jpeg', '*.jpg')))
    image_path_CiB_imag_null_v.set(path_)


# null vector ini doesn't require pois or gau
def getalg_obj():
    pois_or_gau=pois_gau.get()
    return pois_or_gau
    

# for blind ptycho clear all selection
def clear_all_blind_ptycho():
    n_vertical_entry.delete(0,'end')
    overlap_r_entry.delete(0,'end')
    Back_Gd.set('')

    fixed_bd_value.delete(0,'end')
    fixed_bd_label.config(state='disabled')
    fixed_bd_value.config(state='disabled', relief='sunken')

    rank_1.deselect()
    full_rank.deselect()
    rank_per.set(None)
    Perturb_entry.delete(0,'end')
    
    mask_type.set('')

    cordis_entry.delete(0,'end')
    cordis_label.config(state='disabled')
    cordis_entry.config(state='disabled', relief='sunken')

    fresnel_devi_entry.delete(0, 'end')
    fresnel_label.config(state='disabled')
    fresnel_devi_entry.config(state='disabled', relief='sunken')

    image_ty.set(None)
    real_image.deselect()
    com_image.deselect()
    two_image.deselect()
    image_path_real_entry.delete(0,'end')
    image_path_real_entry.config(state='disabled', relief='sunken')
    image_path_real_button.config(state='disabled')

    image_path_rand_entry.delete(0,'end')
    image_path_rand_entry.config(state='disabled', relief='sunken')
    image_path_rand_button.config(state='disabled')

    image_path_CiB_real_entry.delete(0,'end')
    image_path_CiB_imag_entry.delete(0,'end')
    image_path_CiB_real_entry.config(state='disabled', relief='sunken')
    image_path_CiB_real_button.config(state='disabled')
    image_path_CiB_imag_entry.config(state='disabled', relief='sunken')
    image_path_CiB_imag_button.config(state='disabled')


    mask_delta_entry.delete(0,'end')
    MaxIter_entry.delete(0,'end')
    MaxInnerLoop_entry.delete(0, 'end')
    Toler_entry.delete(0,'end')
    os_Rate_entry.delete(0,'end')
    pois_gau.set(None)
    pois_button.deselect()
    gaus_button.deselect()
    gamma_entry.delete(0,'end')
    salt_on_checkbutton.deselect()

    salt_noise_entry.delete(0,'end')
    salt_inten_entry.delete(0,'end')
    salt_noise_entry.config(state='disabled', relief='sunken')
    salt_noise_label.config(state='disabled')
    salt_inten_entry.config(state='disabled', relief='sunken')
    salt_inten_label.config(state='disabled')


# for blind ptycho clear all selection
def clear_all_null_v():
    n_vertical_entry_null_v.delete(0, 'end')
    overlap_r_entry_null_v.delete(0, 'end')
    Back_Gd_null_v.set('')
    fixed_bd_value_null_v.delete(0, 'end')
    rank_1_null_v.deselect()
    full_rank_null_v.deselect()
    rank_per_null_v.set(None)
    Perturb_entry_null_v.delete(0, 'end')

    mask_type_null_v.set('')
    cordis_entry_null_v.delete(0, 'end')
    cordis_label_null_v.config(state='disabled')
    cordis_entry_null_v.config(state='disabled', relief='sunken')

    fresnel_devi_entry_null_v.delete(0, 'end')
    fresnel_label_null_v.config(state='disabled')
    fresnel_devi_entry_null_v.config(state='disabled', relief='sunken')

    image_ty_null_v.set(None)
    real_image_null_v.deselect()
    com_image_null_v.deselect()
    two_image_null_v.deselect()
    image_path_real_entry_null_v.delete(0, 'end')
    image_path_real_entry_null_v.config(state='disabled', relief='sunken')
    image_path_real_button_null_v.config(state='disabled')

    image_path_rand_entry_null_v.delete(0, 'end')
    image_path_rand_entry_null_v.config(state='disabled', relief='sunken')
    image_path_rand_button_null_v.config(state='disabled')

    image_path_CiB_real_entry_null_v.delete(0, 'end')
    image_path_CiB_imag_entry_null_v.delete(0, 'end')
    image_path_CiB_real_entry_null_v.config(state='disabled', relief='sunken')
    image_path_CiB_real_button_null_v.config(state='disabled')
    image_path_CiB_imag_entry_null_v.config(state='disabled', relief='sunken')
    image_path_CiB_imag_button_null_v.config(state='disabled')

    MaxIter_entry_null_v.delete(0, 'end')
    Toler_entry_null_v.delete(0, 'end')
    os_Rate_entry_null_v.delete(0, 'end')
    tau_entry_null_v.delete(0,'end')
    salt_on_checkbutton_null_v.deselect()

    salt_noise_entry_null_v.delete(0, 'end')
    salt_inten_entry_null_v.delete(0, 'end')
    salt_noise_entry_null_v.config(state='disabled', relief='sunken')
    salt_noise_label_null_v.config(state='disabled')
    salt_inten_entry_null_v.config(state='disabled', relief='sunken')
    salt_inten_label_null_v.config(state='disabled')


def clean_data():
    edit_text.delete(1.0, 'end')

def clean_data_null_v():
    edit_text_null_v.delete(1.0, 'end')


# for blind ptychography
def enable_salt():
    if salt_on.get() != 1:
        salt_noise_entry.config(state='disabled', relief='sunken')
        salt_noise_label.config(state='disabled')
        salt_inten_entry.config(state='disabled', relief='sunken')
        salt_inten_label.config(state='disabled')
    if salt_on.get() == 1:
        salt_noise_entry.config(state='normal',relief='solid')
        salt_noise_label.config(state='normal')
        salt_inten_entry.config(state='normal', relief='solid')
        salt_inten_label.config(state='normal')

# for null vector
def enable_salt_null_v():
    if salt_on_null_v.get() != 1:
        salt_noise_entry_null_v.config(state='disabled', relief='sunken')
        salt_noise_label_null_v.config(state='disabled')
        salt_inten_entry_null_v.config(state='disabled', relief='sunken')
        salt_inten_label_null_v.config(state='disabled')
    if salt_on_null_v.get() == 1:
        salt_noise_entry_null_v.config(state='normal',relief='solid')
        salt_noise_label_null_v.config(state='normal')
        salt_inten_entry_null_v.config(state='normal', relief='solid')
        salt_inten_label_null_v.config(state='normal')


# for blind ptycho, import previous config
def import_param():
    # generate workspace path
    workspace_path = os.getcwd()
    initialdir = os.path.join(workspace_path, 'workspace')
    path_ = filedialog.askopenfilename(initialdir=initialdir,title='Please Select "config_bp"',filetypes = (('Text files','*.txt'),))
    pulldata = open(path_).read()
    config_param = pulldata.split('\n')
    input_from_import = {}
    for line in config_param:
        if len(line) > 1:
            param_key, param_value = line.split('=')
            input_from_import[param_key] = param_value
    n_v = input_from_import['n_vertical']
    n_vertical_entry.insert(0, n_v)

    OLR = input_from_import['overlap_r']
    overlap_r_entry.insert(0, OLR)

    bg = input_from_import['background']
    bg_value.set(bg)

    if bg == 'Fixed':
        fixed_bd_v = input_from_import['fixed_bd_value']
        fixed_bd_label.config(state='normal')
        fixed_bd_value.config(state='normal', relief='solid')
        fixed_bd_value.insert(0, fixed_bd_v)

    rank = input_from_import['rank']
    rank_per.set(rank)

    perturb = input_from_import['perturb']
    Perturb.set(perturb)


    mask_type = input_from_import['mask_type']
    mask_type_value.set(mask_type)
    if mask_type == 'Correlated':

        cordist = input_from_import['cordis']
        cordis_label.config(state='normal')
        cordis_entry.config(state='normal', relief='solid')
        cordis.set(cordist)

    elif mask_type == 'Fresnel':

        fresdevi = input_from_import['fresdevi']
        fresnel_label.config(state='normal')
        fresnel_devi_entry.config(state='normal', relief='solid')
        fresnel_devi.set(fresdevi)

    image_type = input_from_import['image_type']
    image_ty.set(image_type)
    if image_type == 'real':
        image_path = input_from_import['image_path']

        image_path_real_entry.config(state='normal', relief='solid')
        image_path_real_button.config(state='normal')
        image_path_real.set(image_path)


    elif image_type =='rand_phase':
        image_path = input_from_import['image_path']

        image_path_rand_entry.config(state='normal', relief='solid')
        image_path_rand_button.config(state='normal')
        image_path_rand.set(image_path)


    else: # CiB case
        image_path_CiBreal = input_from_import['real_path']
        image_path_CiBimag = input_from_import['imag_path']

        image_path_CiB_real_entry.config(state='normal', relief='solid')
        image_path_CiB_real_button.config(state='normal')
        image_path_CiB_imag_entry.config(state='normal', relief='solid')
        image_path_CiB_imag_button.config(state='normal')
        image_path_CiB_real.set(image_path_CiBreal)
        image_path_CiB_imag.set(image_path_CiBimag)

    mask_dlt =input_from_import['mask_delta']
    mask_delta.set(mask_dlt)

    MaxIt = input_from_import['MaxIter']
    MaxIter.set(MaxIt)

    MaxInner = input_from_import['MaxInner']
    MaxInnerLoop.set(MaxInner)

    Tol = input_from_import['Toler']
    Toler.set(Tol)

    os_rate = input_from_import['os_rate']
    os_Rate.set(os_rate)

    pois_or_gau =input_from_import['pois_or_gau']
    pois_gau.set(pois_or_gau)

    ga = input_from_import['gamma']
    gamma.set('ga')

    salton = int(input_from_import['salt_on'])
    salt_on.set(salton)

    if salton == 1:

        salt_prob = input_from_import['salt_prob']
        salt_noise_entry.config(state='normal', relief='solid')
        salt_noise_label.config(state='normal')
        salt_noise.set(salt_prob)

        salt_intensity = input_from_import['salt_inten']
        salt_inten_entry.config(state='normal', relief='solid')
        salt_inten_label.config(state='normal')
        salt_inten.set(salt_intensity)


# for null vector, import previous config
def import_param_null_v():
    # generate workspace path
    workspace_path = os.getcwd()
    initialdir = os.path.join(workspace_path, 'workspace')
    path_ = filedialog.askopenfilename(initialdir=initialdir, title='Please Select "config_nv"',
                                       filetypes=(('Text files', '*.txt'),))
    pulldata = open(path_).read()
    config_param = pulldata.split('\n')
    input_from_import = {}
    for line in config_param:
        if len(line) > 1:
            param_key, param_value = line.split('=')
            input_from_import[param_key] = param_value
    n_v = input_from_import['n_vertical']
    n_vertical_entry_null_v.insert(0, n_v)

    OLR = input_from_import['overlap_r']
    overlap_r_entry_null_v.insert(0, OLR)

    bg = input_from_import['background']
    bg_value_null_v.set(bg)

    # don't consider fixed/ dark boundary at this moment
    # if bg == 'Fixed':
    #     fixed_bd_v = input_from_import['fixed_bd_value']
    #     fixed_bd_label_null_v.config(state='normal')
    #     fixed_bd_value_null_v.config(state='normal', relief='solid')
    #     fixed_bd_value_null_v.insert(0, fixed_bd_v)

    rank = input_from_import['rank']
    rank_per_null_v.set(rank)

    perturb = input_from_import['perturb']
    Perturb_null_v.set(perturb)

    mask_type = input_from_import['mask_type']
    mask_type_value_null_v.set(mask_type)
    if mask_type == 'Correlated':

        cordist = input_from_import['cordis']
        cordis_label_null_v.config(state='normal')
        cordis_entry_null_v.config(state='normal', relief='solid')
        cordis_null_v.set(cordist)

    elif mask_type == 'Fresnel':

        fresdevi = input_from_import['fresdevi']
        fresnel_label_null_v.config(state='normal')
        fresnel_devi_entry_null_v.config(state='normal', relief='solid')
        fresnel_devi_null_v.set(fresdevi)

    image_type = input_from_import['image_type']
    image_ty_null_v.set(image_type)
    if image_type == 'real':
        image_path = input_from_import['image_path']

        image_path_real_entry_null_v.config(state='normal', relief='solid')
        image_path_real_button_null_v.config(state='normal')
        image_path_real_null_v.set(image_path)


    elif image_type == 'rand_phase':
        image_path = input_from_import['image_path']

        image_path_rand_entry_null_v.config(state='normal', relief='solid')
        image_path_rand_button_null_v.config(state='normal')
        image_path_rand_null_v.set(image_path)


    else:  # CiB case
        image_path_CiBreal = input_from_import['real_path']
        image_path_CiBimag = input_from_import['imag_path']

        image_path_CiB_real_entry_null_v.config(state='normal', relief='solid')
        image_path_CiB_real_button_null_v.config(state='normal')
        image_path_CiB_imag_entry_null_v.config(state='normal', relief='solid')
        image_path_CiB_imag_button_null_v.config(state='normal')
        image_path_CiB_real_null_v.set(image_path_CiBreal)
        image_path_CiB_imag_null_v.set(image_path_CiBimag)


    MaxIt = input_from_import['MaxIter']
    MaxIter_null_v.set(MaxIt)

    Tol = input_from_import['Toler']
    Toler_null_v.set(Tol)

    os_rate = input_from_import['os_rate']
    os_Rate_null_v.set(os_rate)

    salton = int(input_from_import['salt_on'])
    salt_on_null_v.set(salton)

    if salton == 1:
        salt_prob = input_from_import['salt_prob']
        salt_noise_entry_null_v.config(state='normal', relief='solid')
        salt_noise_label_null_v.config(state='normal')
        salt_noise_null_v.set(salt_prob)

        salt_intensity = input_from_import['salt_inten']
        salt_inten_entry_null_v.config(state='normal', relief='solid')
        salt_inten_label_null_v.config(state='normal')
        salt_inten_null_v.set(salt_intensity)



# initialize work folder null_vec_folder and blind_ptycho folder
today_date = datetime.datetime.now().strftime('%Y-%m-%d')
def mkdir(savedata_path_parent):

    savedata_path = os.path.join(savedata_path_parent, today_date, '1')
    folder = os.path.exists(savedata_path)
    count = 1
    while folder:
        count += 1
        savedata_path = os.path.join(savedata_path_parent, today_date, str(count))
        folder = os.path.exists(savedata_path)
    os.makedirs(savedata_path)  # if folder number count doesn't exist, create it
    savefig_path = os.path.join(savedata_path, 'fig')
    os.makedirs(savefig_path)
    print('workspace path: %s created' % savedata_path)

    return savedata_path


# for blind ptycho
def run_ptycho():
    get_maxiter_blind_ptycho()
    threading.Thread(target=start_run_blind_ptycho, name='Thread-main').start()

# for null vector
def run_null_v():
    get_maxiter_null_v()
    threading.Thread(target=start_run_null_v, name='Thread-main').start()


# initialize data recorder
# blind ptycho
Iter_blind_ptycho = []
res_y_blind_ptycho = []
rel_im_blind_ptycho = []
rel_mask_blind_ptycho = []
recon_im_blind_ptycho = np.zeros((2, 2))
Iter_lim_blind_ptycho = 2

# null vector
Iter_null_v = []
rel_im_null_v = []
recon_im_null_v = np.zeros((2, 2))
Iter_lim_null_v = 2


# for blind ptycho, get maxiter which is the epoch limit in error plot
def get_maxiter_blind_ptycho():
    global Iter_lim_blind_ptycho
    Maxii = MaxIter.get()
    Iter_lim_blind_ptycho = int(Maxii)


# for null vector
def get_maxiter_null_v():
    global Iter_lim_null_v
    Maxii = MaxIter_null_v.get()
    Iter_lim_null_v = int(Maxii)


# save all parameters to textfile and start running blind ptychography
def start_run_blind_ptycho():
    #
    global recon_im_blind_ptycho, Iter_blind_ptycho, \
        res_y_blind_ptycho, rel_im_blind_ptycho, rel_mask_blind_ptycho

    # clean out the data collect during the previous experiment
    Iter_blind_ptycho = []
    res_y_blind_ptycho = []
    rel_im_blind_ptycho = []
    rel_mask_blind_ptycho = []

    savedata_path_parent = os.path.join(os.getcwd(), 'workspace','blind ptycho')
    savedata_path = mkdir(savedata_path_parent)
    #'''
    today_date=datetime.datetime.now().strftime('%Y-%m-%d')
    #generate savespace
    input_parameters = {'savedata_path': savedata_path}
    n_v = n_vertical_entry.get()
    input_parameters['n_vertical']= n_v

    OLR = overlap_r_entry.get()
    input_parameters['overlap_r'] = OLR

    bg = bg_value.get()
    input_parameters['bg_value'] = bg

    if bg == 'Fixed':
        fixed_bd_v = fixed_bd_value.get()
        input_parameters['fixed_bd_value'] = fixed_bd_v

    rank = getrank()
    input_parameters['rank']=rank

    perturb = Perturb.get()
    input_parameters['perturb'] = perturb

    mask_type = mask_type_value.get()
    input_parameters['mask_type']=mask_type

    if mask_type == 'Correlated':
        cordist = cordis_entry.get()
        input_parameters['cordist'] = cordist
    elif mask_type == 'Fresnel':
        fresdevi = fresnel_devi_entry.get()
        input_parameters['fresdevi'] = fresdevi

    image_type = getimage_t()
    input_parameters['image_type'] = image_type

    if image_type == 'real':
        image_path = image_path_real.get()
        input_parameters['image_path'] = image_path
    elif image_type == 'rand_phase':
        image_path = image_path_rand.get()
        input_parameters['image_path'] = image_path
    elif image_type == 'CiB_image':
        image_path = {'real': image_path_CiB_real.get(),
                      'imag': image_path_CiB_imag.get()}
        input_parameters['image_path_real'] = image_path['real']
        input_parameters['image_path_imag'] = image_path['imag']


    #### get algorithm parameters
    mask_dlt= mask_delta.get()
    input_parameters['mask_delta'] = mask_dlt

    MaxIt= MaxIter.get()
    input_parameters['MaxIter'] = MaxIt

    MaxInner = MaxInnerLoop.get()
    input_parameters['MaxInner'] = MaxInner

    Tol=Toler.get()
    input_parameters['Toler'] = Tol

    os_rate=os_Rate.get()
    input_parameters['os_rate'] = os_rate

    pois_or_gau = pois_gau.get()
    input_parameters['pois_gau'] = pois_or_gau

    ga = gamma.get()
    input_parameters['gamma'] = ga

    salton = salt_on.get()
    input_parameters['salt_on'] = salton

    if salton == 1:
        salt_prob = salt_noise.get()
        input_parameters['salt_prob'] = salt_prob
        salt_intensity = salt_inten.get()
        input_parameters['salt_inten'] = salt_intensity

    with open(os.path.join(savedata_path, 'config_bp.txt'),'w') as f:
        f.write('n_vertical=%s\n' % n_v)
        f.write('overlap_r=%s\n' % OLR)
        f.write('background=%s\n' % bg)  # input_parameters['bg_value']
        if bg == 'Fixed':
            f.write('fixed_bd_value=%s\n' % fixed_bd_v)
        f.write('rank=%s\n' % rank)
        f.write('perturb=%s\n' % perturb)

        f.write('mask_type=%s\n' % mask_type)
        if mask_type == 'Correlated':
            f.write('cordis=%s\n' % cordist)

        elif mask_type == 'Fresnel':
            f.write('fresdevi=%s\n' % fresdevi)

        f.write('image_type=%s\n' % image_type)
        if image_type =='CiB_image':
            f.write('real_path={}\nimag_path={}\n'.format(image_path['real'], image_path['imag']))
        else:
            f.write('image_path={}\n'.format(image_path))

        f.write('mask_delta=%s\n' % mask_dlt)
        f.write('MaxIter=%s\n' % MaxIt)
        f.write('MaxInner=%s\n' % MaxInner)
        f.write('Toler=%s\n' % Tol)
        f.write('os_rate=%s\n' % os_rate)
        f.write('pois_gau=%s\n' % pois_or_gau)
        f.write('gamma=%s\n' % ga)
        f.write('salt_on=%s\n' % salton)
        if salton == 1:
            f.write('salt_prob=%s\n' % salt_prob)
            f.write('salt_inten=%s\n'% salt_intensity)

    '''
    input_parameters = {'n_horizontal':6, 'n_vertical': 6, 'overlap_r': 0.5,
                'bg_value': 'Periodic',
                'fixed_bd_value': 20,
                'rank': 'one',
                'perturb': 2,
                'mask_type': 'IID',
                'image_type': 'CiB_image',
                'image_path_real': '/Users/beaux/Documents/phase retrieval/image_lib/Cameraman.png',
                'image_path_imag': '/Users/beaux/Documents/phase retrieval/image_lib/Cameraman.png',
                'mask_delta': 0.1,
                'MaxIter': 15,
                'MaxInner': 30,
                'Toler': 0.00001,
                'os_rate': 2,
                'gamma': 1,
                'salt_on': 0,
                'salt_prob': 0.01,
                'salt_inten':0.2,
                'pois_gau':'poisson',
                'savedata_path': savedata_path
                        } '''
    print(input_parameters)
    experi = blind_ptycho(input_parameters)
    res_x = 1
    count_DR = 1
    Tol = input_parameters['Toler']
    Max_It = input_parameters['MaxIter']
    #image_type = input_parameters['image_type']
    while res_x > float(Tol) and count_DR < int(Max_It):
        res_x, rel_LPS_x, rel_LPS_mask, count_DR, rec_im = experi.one_step()

        Iter_blind_ptycho.append(count_DR-1) # the output of count_DR is count_DR += 1
        res_y_blind_ptycho.append(res_x)
        rel_im_blind_ptycho.append(rel_LPS_x)
        rel_mask_blind_ptycho.append(rel_LPS_mask)
        blind_ptycho_data_one_step = 'epoch={}: image error={:.4e}, mask error={:.4e} residual={:.4e} \n'.format(count_DR, rel_LPS_x, rel_LPS_mask, res_x)
        edit_text.insert('insert', blind_ptycho_data_one_step)

        if image_type =='real':
            recon_im_blind_ptycho = np.abs(rec_im)
        elif image_type == 'rand_phase':
            recon_im_blind_ptycho = np.abs(rec_im)
        else:
            recon_im_blind_ptycho = np.real(rec_im)

    savefig_path = os.path.join(savedata_path, 'fig')
    savefigs(Iter_blind_ptycho, res_y_blind_ptycho, rel_im_blind_ptycho, rel_mask_blind_ptycho, recon_im_blind_ptycho, savefig_path)
    saved_message = 'figs saved to {}\n \n'.format(savefig_path)
    edit_text.insert('insert', saved_message)
 #
 #
 #
 #
 #
 # save all parameters to textfile and start running null_vector
 #
 #
 #
 #
 #
 #
 #


def start_run_null_v():
    #
    global recon_im_null_v, Iter_null_v, rel_im_null_v

    # clean out the data collect during the previous experiment
    Iter_null_v = []
    rel_im_null_v = []


    savedata_path_parent = os.path.join(os.getcwd(), 'workspace', 'null vector')
    savedata_path = mkdir(savedata_path_parent)
    # '''
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    # generate savespace
    input_parameters = {'savedata_path': savedata_path}
    n_v = n_vertical_entry_null_v.get()
    input_parameters['n_vertical'] = n_v

    OLR = overlap_r_entry_null_v.get()
    input_parameters['overlap_r'] = OLR

    bg = bg_value_null_v.get()
    input_parameters['bg_value'] = bg

    if bg == 'Fixed':
        fixed_bd_v = fixed_bd_value_null_v.get()
        input_parameters['fixed_bd_value'] = fixed_bd_v

    rank = getrank_null_v()
    input_parameters['rank'] = rank

    perturb = Perturb_null_v.get()
    input_parameters['perturb'] = perturb

    mask_type = mask_type_value_null_v.get()
    input_parameters['mask_type'] = mask_type
    if mask_type == 'Correlated':
        cordist = cordis_entry_null_v.get()
        input_parameters['cordist'] = cordist
    elif mask_type == 'Fresnel':
        fresdevi = fresnel_devi_entry_null_v.get()
        input_parameters['fresdevi'] = fresdevi

    image_type = getimage_t_null_v()
    input_parameters['image_type'] = image_type

    if image_type == 'real':
        image_path = image_path_real_null_v.get()
        input_parameters['image_path'] = image_path
    elif image_type == 'rand_phase':
        image_path = image_path_rand_null_v.get()
        input_parameters['image_path'] = image_path
    elif image_type == 'CiB_image':
        image_path = {'real': image_path_CiB_real_null_v.get(),
                      'imag': image_path_CiB_imag_null_v.get()}
        input_parameters['image_path_real'] = image_path['real']
        input_parameters['image_path_imag'] = image_path['imag']

    #### get algorithm parameters

    MaxIt = MaxIter_null_v.get()
    input_parameters['MaxIter'] = MaxIt

    Tol = Toler_null_v.get()
    input_parameters['Toler'] = Tol

    os_rate = os_Rate_null_v.get()
    input_parameters['os_rate'] = os_rate

    tau = tau_null_v.get()
    input_parameters['tau']=tau

    salton = salt_on_null_v.get()
    input_parameters['salt_on'] = salton

    if salton == 1:
        salt_prob = salt_noise_null_v.get()
        input_parameters['salt_prob'] = salt_prob
        salt_intensity = salt_inten_null_v.get()
        input_parameters['salt_inten'] = salt_intensity

    with open(os.path.join(savedata_path, 'config_nv.txt'), 'w') as f:
        f.write('n_vertical=%s\n' % n_v)
        f.write('overlap_r=%s\n' % OLR)
        f.write('background=%s\n' % bg)
        if bg == 'Fixed':
            f.write('fixed_bd_value=%s\n' % fixed_bd_v)
        f.write('rank=%s\n' % rank)
        f.write('perturb=%s\n' % perturb)

        f.write('mask_type=%s\n' % mask_type)

        if mask_type == 'Correlated':
            f.write('cordis=%s\n' % cordist)

        elif mask_type == 'Fresnel':
            f.write('fresdevi=%s\n' % fresdevi)

        f.write('image_type=%s\n' % image_type)
        if image_type == 'CiB_image':
            f.write('real_path={}\nimag_path={}\n'.format(image_path['real'], image_path['imag']))
        else:
            f.write('image_path={}\n'.format(image_path))

        f.write('MaxIter=%s\n' % MaxIt)
        f.write('Toler=%s\n' % Tol)
        f.write('os_rate=%s\n' % os_rate)
        f.write('tau=%s\n' % tau)
        f.write('salt_on=%s \n' % salton)
        if salton == 1:
            f.write('salt_prob=%s\n' % salt_prob)
            f.write('salt_inten=%s\n' % salt_intensity)
    '''
    input_parameters = {'n_horizontal':6, 'n_vertical': 6, 'overlap_r': 0.5,
                'bg_value': 'Periodic',
                'fixed_bd_value': 20,
                'rank': 'one',
                'perturb': 2,
                'mask_type': 'IID',
                'image_type': 'CiB_image',
                'image_path_real': '/Users/beaux/Documents/phase retrieval/image_lib/Cameraman.png',
                'image_path_imag': '/Users/beaux/Documents/phase retrieval/image_lib/Cameraman.png',
                'mask_delta': 0.1,
                'MaxIter': 15,
                'Toler': 0.00001,
                'os_rate': 2,
                'salt_on': 0,
                'salt_prob': 0.01,
                'salt_inten':0.2,
                'savedata_path': savedata_path
                        } '''
    experi_null_v= null_vector_ptycho(input_parameters)
    res_x = 1
    count_DR = 1
    Tol = input_parameters['Toler']
    Max_It = input_parameters['MaxIter']
    # image_type = input_parameters['image_type']
    while res_x > float(Tol) and count_DR < int(Max_It):

        count_DR, rel_im, x_t = experi_null_v.one_step()

        Iter_null_v.append(count_DR)  # the output of count_DR is count_DR += 1
        rel_im_null_v.append(rel_im)
        null_v_data_one_step = 'epoch={}: image error={:.4e} \n'.format(
            count_DR, rel_im)
        edit_text_null_v.insert('insert', null_v_data_one_step)

        if image_type == 'real':
            recon_im_null_v = np.abs(x_t)
        elif image_type == 'rand_phase':
            recon_im_null_v = np.abs(x_t)
        else:
            recon_im_null_v = np.real(x_t)

    plt.figure(0)
    plt.imshow(recon_im_null_v, cmap='gray')
    plt.colorbar()
    plt.grid(False)
    plt.axis('on')
    plt.title('pert={} patch={} olr={}'.format(perturb, experi_null_v.l_patch, OLR))
    savefig_path = os.path.join(savedata_path, 'fig')
    save_path = os.path.join(savefig_path, 'pert={} patch={} olr={}.png'.format(perturb, experi_null_v.l_patch, OLR))
    plt.savefig(save_path)
    plt.close(0)

    plt.figure(1)
    plt.semilogy(Iter_null_v, rel_im_null_v, 'k--', label='image rel err')
    plt.grid(True)
    plt.xlabel('iter')
    plt.ylabel('error')
    #plt.yticks(np.linspace(min(rel_im_null_v), max(rel_im_null_v) + 1, 6))
    plt.title('error plot')
    save_path = os.path.join(savefig_path, 'error plot.png')
    plt.savefig(save_path)
    plt.close(1)

    saved_message = 'figs saved to {}\n \n'.format(savefig_path)
    edit_text_null_v.insert('insert', saved_message)


#######################################################################    
    
'''Start to construct panel '''
# Architecture:
# PR_panel: Tk.window
#       |  page_blind_ptycho: blind ptychography tab
#             | param_frame:
#             | image_plot_frame:
#             | text_area:
#       |  page_null_vector: null vector tab
#             | param_frame_null_v:
#             | image_plot_frame_null_v:
#             | text_area_null_v:
#
#
#


PR_panel = tk.Tk()
PR_panel.title('PhaseReTr_Simulater')



nb = ttk.Notebook(PR_panel)
nb.grid(row=1, column = 0, columnspan = 40, rowspan = 39, sticky = 'NESW')
# blind ptycho experiment
page_blind_ptycho = ttk.Frame(nb)
nb.add(page_blind_ptycho, text = 'blind ptycho')
# null vector experiment
page_null_vector = ttk.Frame(nb)
nb.add(page_null_vector, text = 'null vector')

panel_width = 1100
panel_height = 800

screenwidth = PR_panel.winfo_screenwidth()  
screenheight = PR_panel.winfo_screenheight()  
size = '%dx%d+%d+%d' % (panel_width, panel_height, (screenwidth - panel_width)/2, (screenheight - panel_height)/2)
PR_panel.geometry(size) # width*height + pos_x + pos_y



# blind ptychography

# enter parameters of blind ptychography
param_frame = tk.Frame(page_blind_ptycho)
param_frame.grid(row = 0, column=0, columnspan =3, sticky='w')

# plot result of experiment
image_plot_frame = tk.Frame(page_blind_ptycho, width=100, height=120, background="bisque")
image_plot_frame.grid(row = 1, column = 0, columnspan=2, sticky='w')

# show errors text widget
text_area = tk.Frame(master=page_blind_ptycho)
text_area.grid(row=1, column=2, columnspan=1, sticky='wn')
text_area_label_frame = tk.LabelFrame(text_area, width=75, height=20, text='show data')
text_area_label_frame.grid(row = 0, column = 0, rowspan = 3, columnspan=10, sticky = 'wnes')

#blind_ptycho_data = 'kkkkk'
edit_text = tkst.ScrolledText(master=text_area_label_frame, wrap=tk.WORD, width=60, height=15)
edit_text.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)
#edit_text.insert('insert', blind_ptycho_data)

# clean text data prevent from crashing
clean_text_data=tk.Button(text_area, text="clean data", command=clean_data, relief='solid')
clean_text_data.grid(row=3, column = 8, sticky='n')
clean_text_data.config(height = 2, width = 9)


####
#### SCAN TYPE #### 
####
SCAN_TYPE = tk.LabelFrame(param_frame, width = 96, height =96, text = 'SCAN TYPE')
SCAN_TYPE.grid(row=1, column = 0, padx=6, sticky = 'w')

n_vertical_label=tk.Label(SCAN_TYPE, text= 'N_V')
n_vertical_label.grid(row=0,column=0,sticky='w')
n_vertical_entry = tk.Entry(SCAN_TYPE, relief = 'solid', width = 3)
n_vertical_entry.grid(row=0,column=1)
# tip button
gethelp_nv = tk.Button(SCAN_TYPE,text='tip', command=gettipsnv).grid(row=0, column=2)



#n_horizontal_label=tk.Label(SCAN_TYPE, text= 'N_H')
#n_horizontal_label.grid(row=1,column=0,sticky='w')
#n_horizontal_entry = tk.Entry(SCAN_TYPE, relief = 'solid', width = 3)
#n_horizontal_entry.grid(row=1,column=1)
# tip button
#gethelp_nh = tk.Button(SCAN_TYPE,text='tip', command=gettipsnh).grid(row=1, column=2)
#### 

overlap_r=tk.Label(SCAN_TYPE, text= 'OLR')
overlap_r.grid(row=2,column=0,sticky='w')
overlap_r_entry = tk.Entry(SCAN_TYPE, relief = 'solid', width = 3)
overlap_r_entry.grid(row=2, column=1)
gethelp_olr = tk.Button(SCAN_TYPE,text='tip', command=gettipsolr).grid(row=2, column=2)
####

bg_value=tk.StringVar()
Back_Gd_label=tk.Label(SCAN_TYPE,text='Bakc_Gd')
Back_Gd_label.grid(row=3,column=0, sticky='w')
Back_Gd=ttk.Combobox(SCAN_TYPE,width=8,textvariable=bg_value)
Back_Gd['values']=('Periodic','Dark','Fixed')

Back_Gd.bind("<<ComboboxSelected>>",bg_selection)
Back_Gd.grid(row=3,column=1,columnspan=1,sticky='w')

fixed_bd_label= tk.Label(SCAN_TYPE, text='intensity',state='disabled')
fixed_bd_value= tk.Entry(SCAN_TYPE, width = 3,state='disabled')
fixed_bd_label.grid(row=3, column=2)
fixed_bd_value.grid(row=3, column=3)
####
# rank 1 verse full rank
####
rank_per= tk.StringVar()
rank_per.set(' ')
rank_1 = tk.Radiobutton(SCAN_TYPE, text= 'rank one', variable = rank_per, value = 'one', command= getrank)
rank_1.grid(row=4, column=0, sticky='w')
full_rank = tk.Radiobutton(SCAN_TYPE, text= 'full rank', variable = rank_per, value = 'full', command= getrank)
full_rank.grid(row=4, column=2,sticky='w')


Perturb = tk.StringVar()
tk.Label(SCAN_TYPE, text='PertB').grid(row=5, column=0, sticky='w')
Perturb_entry = tk.Entry(SCAN_TYPE, textvariable=Perturb,relief='solid',width=3)
Perturb_entry.grid(row=5,column=1)


####
####  mask property
#### 
MASKPROP_TYPE = tk.LabelFrame(param_frame, width = 96, height =96, text = 'MASK PROPERTY')
MASKPROP_TYPE.grid(row=1, column = 1,padx = 6, sticky = 'wens')
#### mask_type

mask_type_value=tk.StringVar()
mask_type_label=tk.Label(MASKPROP_TYPE, text='mask type')
mask_type_label.grid(row=0, column=0, sticky='w')
mask_type = ttk.Combobox(MASKPROP_TYPE,width=15,textvariable=mask_type_value)
mask_type['values']=('Fresnel','Correlated','IID')
mask_type.bind("<<ComboboxSelected>>", enable_mask_blind_ptycho)
mask_type.grid(row=0,column=1,padx=3,pady=3,sticky='w')

# add correlate distance.
cordis_label = tk.Label(MASKPROP_TYPE,text='Corr Dist', state='disabled')
cordis_label.grid(row=1, column=0)
cordis = tk.StringVar()
cordis_entry = tk.Entry(MASKPROP_TYPE, textvariable=cordis,state='disabled', width=3)
cordis_entry.grid(row=1,column=1)
gethelp_cordis = tk.Button(MASKPROP_TYPE, text='tip', command=gettipcordist).grid(row=1, column=2)

# add fresnel deviation
fresnel_label = tk.Label(MASKPROP_TYPE,text='Fres Devi', state='disabled')
fresnel_label.grid(row=2, column=0)
fresnel_devi = tk.StringVar()
fresnel_devi_entry = tk.Entry(MASKPROP_TYPE, textvariable=fresnel_devi,state='disabled', width=3)
fresnel_devi_entry.grid(row=2,column=1)
gethelp_fresdevi = tk.Button(MASKPROP_TYPE, text='tip', command=gettipfresdevi).grid(row=2, column=2)


####
####   image type
####
IMAGE_TYPE = tk.LabelFrame(param_frame, width = 160, height =400, text = 'IMAGE TYPE')
IMAGE_TYPE.grid(row=1, column = 2, rowspan=1,padx = 6, sticky = 'wnes')

#### IMAGE real image; image * rand phase, Real+ i Image
image_ty= tk.StringVar()
image_ty.set(' ')
# real_image button
real_image = tk.Radiobutton(IMAGE_TYPE, text= 'real image',variable = image_ty,
                            value = 'real', command= getimage_t)
real_image.grid(row=0, column=0, sticky='w')
# complex_image button
com_image = tk.Radiobutton(IMAGE_TYPE, text= 'rand_phase', variable = image_ty,
                           value = 'rand_phase', command= getimage_t)
com_image.grid(row=1, column=0,sticky='w')
# CiB button
two_image = tk.Radiobutton(IMAGE_TYPE, text = 'CiB_image', variable = image_ty,
                           value = 'CiB_image', command = getimage_t)
two_image.grid(row=2, column=0, sticky='w')

##REAL IMAGE
image_path_real= tk.StringVar() # string location that stores path
image_path_real_entry=tk.Entry(IMAGE_TYPE, textvariable = image_path_real, state='disabled')
image_path_real_entry.grid(row=0, column=1)
image_path_real_button=tk.Button(IMAGE_TYPE,text='choose path', command=selectPath_real,
                                 state='disabled', width=9)
image_path_real_button.grid(row=0, column=2)
#RAND IMAGE
image_path_rand= tk.StringVar()
image_path_rand_entry=tk.Entry(IMAGE_TYPE, textvariable = image_path_rand, state='disabled')
image_path_rand_entry.grid(row=1, column=1)
image_path_rand_button=tk.Button(IMAGE_TYPE,text='choose path', command=selectPath_rand,
                                 state='disabled', width=9)
image_path_rand_button.grid(row=1, column=2)
#CIB
image_path_CiB_real= tk.StringVar()
image_path_CiB_real_entry=tk.Entry(IMAGE_TYPE, textvariable = image_path_CiB_real,state='disabled')
image_path_CiB_real_entry.grid(row=2, column=1)
image_path_CiB_real_button=tk.Button(IMAGE_TYPE,text='real part', command=selectPath_CiB_real,
                                     state='disabled', width=9)
image_path_CiB_real_button.grid(row=2, column=2)
#2nd of CIB
image_path_CiB_imag= tk.StringVar()
image_path_CiB_imag_entry=tk.Entry(IMAGE_TYPE, textvariable = image_path_CiB_imag, state='disabled')
image_path_CiB_imag_entry.grid(row=3, column=1)
image_path_CiB_imag_button=tk.Button(IMAGE_TYPE,text='imag part', command=selectPath_CiB_imag,
                                     state='disabled', width=9)
image_path_CiB_imag_button.grid(row=3, column=2)


####
#### algorithm parameters
####
ALG_SET = tk.LabelFrame(param_frame, width = 96, height =96, text = 'ALGORITHM PARAMETER')
ALG_SET.grid(row=2, column = 0,padx = 6, sticky = 'wnes')


tk.Label(ALG_SET, text='mask delta').grid(row=0, column=0, sticky='w')
mask_delta=tk.StringVar()
mask_delta_entry = tk.Entry(ALG_SET, textvariable=mask_delta,relief='solid',width=3)
mask_delta_entry.grid(row=0,column=1)

tk.Label(ALG_SET, text='MaxIter').grid(row=1, column=0, sticky='w')
MaxIter = tk.StringVar()
MaxIter_entry = tk.Entry(ALG_SET, textvariable=MaxIter, relief='solid',width=3)
MaxIter_entry.grid(row=1, column=1)

tk.Label(ALG_SET, text='MaxInner').grid(row=1, column=2, sticky='w')
MaxInnerLoop=tk.StringVar()
MaxInnerLoop.set('30')
MaxInnerLoop_entry = tk.Entry(ALG_SET, textvariable=MaxInnerLoop, relief='solid',width=3)
MaxInnerLoop_entry.grid(row=1, column=3)

tk.Label(ALG_SET, text='Toler').grid(row=2, column=0, sticky='w')
Toler=tk.StringVar()
Toler_entry = tk.Entry(ALG_SET, textvariable=Toler, relief='solid',width=3)
Toler_entry.grid(row=2, column=1)

tk.Label(ALG_SET, text='os_rate').grid(row=3, column=0, sticky='w')
os_Rate=tk.StringVar()
os_Rate_entry = tk.Entry(ALG_SET, textvariable=os_Rate,relief='solid',width=3)
os_Rate_entry.grid(row=3,column=1)


pois_gau= tk.StringVar()
pois_gau.set(' ')
pois_button = tk.Radiobutton(ALG_SET, text= 'poisson', variable = pois_gau, value = 'poisson', command= getalg_obj)
pois_button.grid(row=4, column=0, sticky='w')
gaus_button = tk.Radiobutton(ALG_SET, text= 'gaussian', variable = pois_gau, value = 'gaussian', command= getalg_obj)
gaus_button.grid(row=4, column=1,sticky='w')


gamma=tk.StringVar()
tk.Label(ALG_SET, text='gamma').grid(row=5, column=0, sticky='w')
gamma_entry = tk.Entry(ALG_SET, textvariable=gamma, relief='solid', width=3)
gamma_entry.grid(row=5, column=1)


####
#### NOISE triggered
####

NOISE_TRIGGER = tk.LabelFrame(param_frame, width = 96, height =96, text = 'Noise Trigger')
NOISE_TRIGGER.grid(row=2, column = 1, rowspan=1,padx = 6, sticky = 'wnes')

salt_on = tk.IntVar()
salt_on_checkbutton = tk.Checkbutton(NOISE_TRIGGER,text='salt enable', variable = salt_on, command=enable_salt)
salt_on_checkbutton.grid(row=0, column=0)

salt_noise_label = tk.Label(NOISE_TRIGGER,text='salt prob', state='disabled')
salt_noise_label.grid(row=1, column=0)
salt_noise = tk.StringVar()
salt_noise_entry = tk.Entry(NOISE_TRIGGER, textvariable=salt_noise,state='disabled', width=3)
salt_noise_entry.grid(row=1,column=1)

salt_inten_label = tk.Label(NOISE_TRIGGER,text='salt inten', state='disabled')
salt_inten_label.grid(row=2, column=0)
salt_inten = tk.StringVar()
salt_inten_entry = tk.Entry(NOISE_TRIGGER, textvariable=salt_inten,state='disabled', width=3)
salt_inten_entry.grid(row=2,column=1)

# import previous experiment
import_param_button=tk.Button(param_frame, text='import parameters', command=import_param, relief='solid')
import_param_button.grid(row=2, column = 2)
import_param_button.config(height = 2, width = 18)

# start button #
start_button_blind_ptycho=tk.Button(param_frame,text='start', command=run_ptycho, relief='solid')
start_button_blind_ptycho.grid(row=3, column = 0, sticky='n')
start_button_blind_ptycho.config(height = 2, width = 7)

# exit button #
exit_button_blind_ptycho=tk.Button(param_frame, text="exit", command = PR_panel.quit, relief='solid')
exit_button_blind_ptycho.grid(row=3, column = 1, sticky='n')
exit_button_blind_ptycho.config(height = 2, width = 7)

# clear all button #
clear_all_button_blind_ptycho=tk.Button(param_frame, text='clear all', relief='solid',command=clear_all_blind_ptycho)
#CLEAR_ALL_BUTTON.bind('<<Button-1>>', clear_all)
clear_all_button_blind_ptycho.grid(row=3, column = 2, sticky='n')
clear_all_button_blind_ptycho.config(height = 2, width = 7)

### matplot live in tkinter

figure_panel = tk.LabelFrame(image_plot_frame, width = 100, height=50, text = 'figure panel')
figure_panel.grid(row = 2, column = 2, rowspan = 1, sticky = 'wnes')
style.use("ggplot")

live_fig = Figure(figsize=(6, 3), dpi=100)
error_plot = live_fig.add_subplot(121)
image_recon = live_fig.add_subplot(122)
live_fig.subplots_adjust(wspace=0.1, hspace=0.3)

canvas = FigureCanvasTkAgg(live_fig, figure_panel)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(canvas, figure_panel)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def animate_plot_blind_ptycho(i):
    global recon_im_blind_ptycho, Iter_blind_ptycho,res_y_blind_ptycho,rel_im_blind_ptycho,\
        rel_mask_blind_ptycho,Iter_lim_blind_ptycho
    image_recon.clear()
    image_recon.imshow(recon_im_blind_ptycho, cmap='gray')
    image_recon.set_axis_off()


    error_plot.clear()
    error_plot.semilogy(Iter_blind_ptycho, res_y_blind_ptycho, 'r-')
    error_plot.semilogy(Iter_blind_ptycho, rel_im_blind_ptycho, 'b:')
    error_plot.semilogy(Iter_blind_ptycho, rel_mask_blind_ptycho, 'k--')
    #error_plot.set_xlim([1, Iter_lim_blind_ptycho])
    #error_plot.set_ylim([1e-8, 1.5])
    error_plot.set_title('Error Plot', size='12')
    error_plot.set_xlabel('epoch', fontsize='10')
    error_plot.xaxis.set_label_coords(1.01, -0.09)
    error_plot.set_ylabel('error', fontsize='10')
    error_plot.legend(('Residual', 'Mask RelErr', 'Imge RelErr'),
                      loc='lower left')


ani = animation.FuncAnimation(live_fig, animate_plot_blind_ptycho, interval=1000)

# end matplot line in tkinter

#
#
#
#
#
#
#
#
#
# null vector; null_v short for null vector
#
#
#
#
#
#
#
#
#
#


# enter parameters of blind ptychography
param_frame_null_v = tk.Frame(page_null_vector)
param_frame_null_v.grid(row = 0, column=0, columnspan =3, sticky='w')

# plot result of experiment
image_plot_frame_null_v = tk.Frame(page_null_vector, width=100, height=120, background="bisque")
image_plot_frame_null_v.grid(row = 1, column = 0, columnspan=2, sticky='w')

# show errors text widget
text_area_null_v = tk.Frame(master=page_null_vector)
text_area_null_v.grid(row=1, column=2, columnspan=1, sticky='wn')
text_area_label_frame_null_v = tk.LabelFrame(text_area_null_v, width=75, height=20, text='show data')
text_area_label_frame_null_v.grid(row = 0, column = 0, rowspan = 3, columnspan=10, sticky = 'wnes')

# showing data text scrolledtext
edit_text_null_v = tkst.ScrolledText(master=text_area_label_frame_null_v, wrap=tk.WORD, width=60, height=15)
edit_text_null_v.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)

# clean text data button
clean_text_data_null_v=tk.Button(text_area_null_v, text="clean data", command=clean_data_null_v, relief='solid')
clean_text_data_null_v.grid(row=3, column = 8, sticky='n')
clean_text_data_null_v.config(height = 2, width = 9)
####
#### SCAN TYPE ####
####
SCAN_TYPE_null_v = tk.LabelFrame(param_frame_null_v, width = 96, height =96, text = 'SCAN TYPE')
SCAN_TYPE_null_v.grid(row=1, column = 0, padx=6, sticky = 'w')

n_vertical_label_null_v=tk.Label(SCAN_TYPE_null_v, text= 'N_V')
n_vertical_label_null_v.grid(row=0,column=0,sticky='w')
n_vertical_entry_null_v = tk.Entry(SCAN_TYPE_null_v, relief = 'solid', width = 3)
n_vertical_entry_null_v.grid(row=0,column=1)
# tip button
gethelp_nv_null_v = tk.Button(SCAN_TYPE_null_v,text='tip', command=gettipsnv).grid(row=0, column=2)

# overlapping ratio

overlap_r_null_v = tk.Label(SCAN_TYPE_null_v, text= 'OLR')
overlap_r_null_v.grid(row=2, column=0, sticky='w')
overlap_r_entry_null_v = tk.Entry(SCAN_TYPE_null_v, relief = 'solid', width = 3)
overlap_r_entry_null_v.grid(row=2, column=1)
gethelp_olr_null_v = tk.Button(SCAN_TYPE_null_v,text='tip', command=gettipsolr).grid(row=2, column=2)
####

bg_value_null_v = tk.StringVar()
Back_Gd_label_null_v=tk.Label(SCAN_TYPE_null_v, text='Bakc_Gd')
Back_Gd_label_null_v.grid(row=3,column=0, sticky='w')
Back_Gd_null_v = ttk.Combobox(SCAN_TYPE_null_v,width=8,textvariable=bg_value_null_v)
#Back_Gd_null_v['values']=('Periodic','Dark','Fixed')
# only consider periodic boundary at this moment
Back_Gd_null_v['values']=('Periodic')

Back_Gd_null_v.bind("<<ComboboxSelected>>",bg_selection_null_v)
Back_Gd_null_v.grid(row=3,column=1,columnspan=1,sticky='w')

fixed_bd_label_null_v = tk.Label(SCAN_TYPE_null_v, text='intensity',state='disabled')
fixed_bd_value_null_v = tk.Entry(SCAN_TYPE_null_v, width = 3,state='disabled')
fixed_bd_label_null_v.grid(row=3, column=2)
fixed_bd_value_null_v.grid(row=3, column=3)
####
# rank 1 verse full rank
####
rank_per_null_v = tk.StringVar()
rank_per_null_v.set(' ')
rank_1_null_v = tk.Radiobutton(SCAN_TYPE_null_v, text='rank one', variable=rank_per_null_v, value='one', command=getrank_null_v)
rank_1_null_v.grid(row=4, column=0, sticky='w')
full_rank_null_v = tk.Radiobutton(SCAN_TYPE_null_v, text='full rank', variable=rank_per_null_v, value='full', command=getrank_null_v)
full_rank_null_v.grid(row=4, column=2,sticky='w')


Perturb_null_v = tk.StringVar()
tk.Label(SCAN_TYPE_null_v, text='PertB').grid(row=5, column=0, sticky='w')
Perturb_entry_null_v = tk.Entry(SCAN_TYPE_null_v, textvariable=Perturb_null_v, relief='solid',width=3)
Perturb_entry_null_v.grid(row=5,column=1)


####
####  mask property
####
MASKPROP_TYPE_null_v = tk.LabelFrame(param_frame_null_v, width = 96, height =96, text = 'MASK PROPERTY')
MASKPROP_TYPE_null_v.grid(row=1, column = 1,padx = 6, sticky = 'wens')
#### mask_type

mask_type_value_null_v = tk.StringVar()
mask_type_label_null_v = tk.Label(MASKPROP_TYPE_null_v, text='mask type')
mask_type_label_null_v.grid(row=0, column=0, sticky='w')
mask_type_null_v = ttk.Combobox(MASKPROP_TYPE_null_v,width=15,textvariable=mask_type_value_null_v)
mask_type_null_v['values']=('Fresnel','Correlated','IID')
mask_type_null_v.bind("<<ComboboxSelected>>",enable_mask_null_v)
mask_type_null_v.grid(row=0,column=1,padx=3,pady=3,sticky='w')

# add correlate distance.
cordis_label_null_v = tk.Label(MASKPROP_TYPE_null_v,text='Corr Dist', state='disabled')
cordis_label_null_v.grid(row=1, column=0)
cordis_null_v = tk.StringVar()
cordis_entry_null_v = tk.Entry(MASKPROP_TYPE_null_v, textvariable=cordis_null_v, state='disabled', width=3)
cordis_entry_null_v.grid(row=1,column=1)
gethelp_cordis_null_v = tk.Button(MASKPROP_TYPE_null_v, text='tip', command=gettipcordist).grid(row=1, column=2)


# add fresnel deviation
fresnel_label_null_v = tk.Label(MASKPROP_TYPE_null_v,text='Fres Devi', state='disabled')
fresnel_label_null_v.grid(row=2, column=0)
fresnel_devi_null_v = tk.StringVar()
fresnel_devi_entry_null_v = tk.Entry(MASKPROP_TYPE_null_v, textvariable=fresnel_devi_null_v,state='disabled', width=3)
fresnel_devi_entry_null_v.grid(row=2,column=1)
gethelp_fresdevi_null_v = tk.Button(MASKPROP_TYPE_null_v, text='tip', command=gettipfresdevi).grid(row=2, column=2)

####
####   image type
####
IMAGE_TYPE_null_v = tk.LabelFrame(param_frame_null_v, width = 160, height =400, text = 'IMAGE TYPE')
IMAGE_TYPE_null_v.grid(row=1, column = 2, rowspan=1,padx = 6, sticky = 'wnes')

#### IMAGE real image; image * rand phase, Real+ i Image
image_ty_null_v= tk.StringVar()
image_ty_null_v.set(' ')
# real_image button
real_image_null_v = tk.Radiobutton(IMAGE_TYPE_null_v, text= 'real image',variable=image_ty_null_v,
                            value = 'real', command= getimage_t_null_v)
real_image_null_v.grid(row=0, column=0, sticky='w')
# complex_image button
com_image_null_v = tk.Radiobutton(IMAGE_TYPE_null_v, text= 'rand_phase', variable=image_ty_null_v,
                           value = 'rand_phase', command= getimage_t_null_v)
com_image_null_v.grid(row=1, column=0,sticky='w')
# CiB button
two_image_null_v = tk.Radiobutton(IMAGE_TYPE_null_v, text = 'CiB_image', variable=image_ty_null_v,
                           value = 'CiB_image', command = getimage_t_null_v)
two_image_null_v.grid(row=2, column=0, sticky='w')

##REAL IMAGE
image_path_real_null_v= tk.StringVar() # string location that stores path
image_path_real_entry_null_v = tk.Entry(IMAGE_TYPE_null_v, textvariable = image_path_real_null_v, state='disabled')
image_path_real_entry_null_v.grid(row=0, column=1)
image_path_real_button_null_v = tk.Button(IMAGE_TYPE_null_v,text='choose path', command=selectPath_real_null_v,
                                 state='disabled', width=9)
image_path_real_button_null_v.grid(row=0, column=2)
#RAND IMAGE
image_path_rand_null_v= tk.StringVar()
image_path_rand_entry_null_v=tk.Entry(IMAGE_TYPE_null_v, textvariable=image_path_rand_null_v, state='disabled')
image_path_rand_entry_null_v.grid(row=1, column=1)
image_path_rand_button_null_v=tk.Button(IMAGE_TYPE_null_v,text='choose path', command=selectPath_rand_null_v,
                                 state='disabled', width=9)
image_path_rand_button_null_v.grid(row=1, column=2)
#CIB
image_path_CiB_real_null_v= tk.StringVar()
image_path_CiB_real_entry_null_v=tk.Entry(IMAGE_TYPE_null_v, textvariable=image_path_CiB_real_null_v, state='disabled')
image_path_CiB_real_entry_null_v.grid(row=2, column=1)
image_path_CiB_real_button_null_v=tk.Button(IMAGE_TYPE_null_v,text='real part', command=selectPath_CiB_real_null_v,
                                     state='disabled', width=9)
image_path_CiB_real_button_null_v.grid(row=2, column=2)
#2nd of CIB
image_path_CiB_imag_null_v = tk.StringVar()
image_path_CiB_imag_entry_null_v = tk.Entry(IMAGE_TYPE_null_v, textvariable=image_path_CiB_imag_null_v, state='disabled')
image_path_CiB_imag_entry_null_v.grid(row=3, column=1)
image_path_CiB_imag_button_null_v = tk.Button(IMAGE_TYPE_null_v, text='imag part', command=selectPath_CiB_imag_null_v,
                                     state='disabled', width=9)
image_path_CiB_imag_button_null_v.grid(row=3, column=2)


####
#### algorithm parameters
####
ALG_SET_null_v = tk.LabelFrame(param_frame_null_v, width = 96, height =96, text = 'ALGORITHM PARAMETER')
ALG_SET_null_v.grid(row=2, column = 0,padx = 6, sticky = 'wnes')


tk.Label(ALG_SET_null_v, text='MaxIter').grid(row=1, column=0, sticky='w')
MaxIter_null_v = tk.StringVar()
MaxIter_entry_null_v = tk.Entry(ALG_SET_null_v, textvariable=MaxIter_null_v, relief='solid',width=3)
MaxIter_entry_null_v.grid(row=1, column=1)


tk.Label(ALG_SET_null_v, text='Toler').grid(row=2, column=0, sticky='w')
Toler_null_v=tk.StringVar()
Toler_entry_null_v = tk.Entry(ALG_SET_null_v, textvariable=Toler_null_v, relief='solid',width=3)
Toler_entry_null_v.grid(row=2, column=1)


tk.Label(ALG_SET_null_v, text='os_rate').grid(row=3, column=0, sticky='w')
os_Rate_null_v = tk.StringVar()
os_Rate_entry_null_v = tk.Entry(ALG_SET_null_v, textvariable=os_Rate_null_v,relief='solid',width=3)
os_Rate_entry_null_v.grid(row=3,column=1)

tk.Label(ALG_SET_null_v, text='tau').grid(row=4, column=0, sticky='w')
tau_null_v = tk.StringVar()
tau_entry_null_v = tk.Entry(ALG_SET_null_v, textvariable=tau_null_v,relief='solid',width=3)
tau_entry_null_v.grid(row=4, column=1)



####
#### NOISE triggered
####

NOISE_TRIGGER_null_v = tk.LabelFrame(param_frame_null_v, width = 96, height =96, text = 'Noise Trigger')
NOISE_TRIGGER_null_v.grid(row=2, column = 1, rowspan=1,padx = 6, sticky = 'wnes')

salt_on_null_v = tk.IntVar()
salt_on_checkbutton_null_v = tk.Checkbutton(NOISE_TRIGGER_null_v, text='salt enable', variable = salt_on_null_v, command=enable_salt_null_v)
salt_on_checkbutton_null_v.grid(row=0, column=0)

salt_noise_label_null_v = tk.Label(NOISE_TRIGGER_null_v, text='salt prob', state='disabled')
salt_noise_label_null_v.grid(row=1, column=0)
salt_noise_null_v = tk.StringVar()
salt_noise_entry_null_v = tk.Entry(NOISE_TRIGGER_null_v, textvariable=salt_noise_null_v, state='disabled', width=3)
salt_noise_entry_null_v.grid(row=1,column=1)

salt_inten_label_null_v = tk.Label(NOISE_TRIGGER_null_v,text='salt inten', state='disabled')
salt_inten_label_null_v.grid(row=2, column=0)
salt_inten_null_v = tk.StringVar()
salt_inten_entry_null_v = tk.Entry(NOISE_TRIGGER_null_v, textvariable=salt_inten_null_v,state='disabled', width=3)
salt_inten_entry_null_v.grid(row=2,column=1)

# import previous experiment
import_param_button_null_v=tk.Button(param_frame_null_v, text='import parameters', command=import_param_null_v, relief='solid')
import_param_button_null_v.grid(row=2, column = 2)
import_param_button_null_v.config(height = 2, width = 18)

# start button #
#start_button=tk.Button(PR_panel,text='start', command=start_run, relief='solid')
start_button_null_v=tk.Button(param_frame_null_v,text='start', command=run_null_v, relief='solid')
start_button_null_v.grid(row=3, column = 0, sticky='n')
start_button_null_v.config(height = 2, width = 7)

# exit button #
exit_button_null_v=tk.Button(param_frame_null_v, text="exit", command=PR_panel.quit, relief='solid')
exit_button_null_v.grid(row=3, column = 1, sticky='n')
exit_button_null_v.config(height = 2, width = 7)
# clear all button #
clear_all_button_null_v=tk.Button(param_frame_null_v, text='clear all', relief='solid',command=clear_all_null_v)
#CLEAR_ALL_BUTTON.bind('<<Button-1>>', clear_all)
clear_all_button_null_v.grid(row=3, column = 2, sticky='n')
clear_all_button_null_v.config(height = 2, width = 7)

### matplot live in tkinter

figure_panel_null_v = tk.LabelFrame(image_plot_frame_null_v, width = 100, height=50, text = 'figure panel')
figure_panel_null_v.grid(row = 2, column = 2, rowspan = 1, sticky = 'wnes')
style.use("ggplot")

live_fig_null_v = Figure(figsize=(6, 3), dpi=100)
error_plot_null_v = live_fig_null_v.add_subplot(121)
image_recon_null_v = live_fig_null_v.add_subplot(122)
live_fig_null_v.subplots_adjust(wspace=0.1, hspace=0.3)

canvas_null_v = FigureCanvasTkAgg(live_fig_null_v, figure_panel_null_v)
canvas_null_v.draw()
canvas_null_v.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

toolbar_null_v = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(canvas_null_v, figure_panel_null_v)
toolbar_null_v.update()
canvas_null_v._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def animate_plot_null_v(i):
    global recon_im_null_v, Iter_null_v, rel_im_null_v, Iter_lim_null_v
    image_recon_null_v.clear()
    image_recon_null_v.imshow(recon_im_null_v, cmap='gray')
    image_recon_null_v.set_axis_off()

    error_plot_null_v.clear()
    error_plot_null_v.semilogy(Iter_null_v, rel_im_null_v, 'k--')
    #error_plot_null_v.set_xlim([1, Iter_lim_null_v])
    #error_plot_null_v.set_ylim([1e-8, 1.5])
    error_plot_null_v.set_title('Error Plot', size='12')
    error_plot_null_v.set_xlabel('iter', fontsize='10')
    error_plot_null_v.xaxis.set_label_coords(1.01, -0.09)
    error_plot_null_v.set_ylabel('error', fontsize='10')
    error_plot_null_v.legend(('Image RelErr',),
                      loc='lower left')


ani_null_v = animation.FuncAnimation(live_fig_null_v, animate_plot_null_v, interval=1000)


PR_panel.mainloop()




















