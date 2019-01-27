#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 19 12:55:15 2018

PYTHON TKINTER GUI For phase retreival ptychography
@author: Zheqing Zhang 
Email: zheqing@math.ucdavis.edu
"""
##### import matplotlib
###
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from blind_ptychography import ptycho


# import threading allowing animate running seperately.
import threading




# import numpy
import numpy as np






##### import tkinter
import tkinter as tk
import tkinter.messagebox as msb
from tkinter import ttk
from tkinter import filedialog


import datetime
import os

#### Class tooltips
class CreateTips(tk.Entry):
    def __init__(self, *args,tip_message , **kwargs):
        tk.Entry.__init__(self, *args, **kwargs)
        self.message=tip_message
        self.l1 = tk.Label(self,text='hover')
        self.l2 = tk.Label(self, text="", width=40)
        self.l1.pack(side="top")
        self.l2.pack(side="top", fill="x")

        self.l1.bind("<Enter>", self.on_enter)
        self.l1.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.l2.configure(text=self.tip_message)

    def on_leave(self, enter):
        self.l2.configure(text="")
#### end class tooltips



class TipButton(tk.Button):
    def __init__(self,*args,tip_m, **kwargs):
        tk.Button.__init__(self, *args, **kwargs)
        self.tip_m=tip_m
        
    def gettips(self):
        tk.Button.gettips(self)
        messagebox.showinfo('新建文件','您已成功创建个人资料文档')




def gettipsnv():
    msb.showinfo('n_veritial', \
                 'Number of small patches you want to split the image into verically.'
                 'Positive integer. Once fixed, the square patch size is determined')

def gettipsnh():
    msb.showinfo('n_horizontal','Number of small patches you want to split the image into horizontally.')

def gettipsolr():
    msb.showinfo('OLR','Overlaping ratio between two adjacent measurements')


def bg_selection(*args):
    
    if bg_value.get() == 'Fixed':
        fixed_bd_label.config(state='normal')
        fixed_bd_value.config(state='normal',relief='solid')
        
    if bg_value.get() != 'Fixed':
        fixed_bd_label.config(state='disabled')
        fixed_bd_value.config(state='disabled', relief='sunken')
    
    

def getrank():
    rank = rank_per.get()
    return rank
    
def getimage_t():
    image_type = image_ty.get()
    #image_path_real.set(' ')
    #image_path_rand.set(' ')
    #image_path_CiB.set(' ')
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


def getalg_obj():
    pois_or_gau=pois_gau.get()
    return pois_or_gau
    
    


###### clear all selection  
def clear_all():
    n_vertical_entry.delete(0,'end')
    #n_horizontal_entry.delete(0,'end')
    overlap_r_entry.delete(0,'end')
    Back_Gd.set('')
    fixed_bd_value.delete(0,'end')
    rank_1.deselect()
    full_rank.deselect()
    rank_per.set(None)
    Perturb_entry.delete(0,'end')
    
    mask_type.set('')
    image_ty.set(None)
    real_image.deselect()
    com_image.deselect()
    two_image.deselect()
    image_path_real_entry.delete(0,'end')
    image_path_rand_entry.delete(0,'end')
    image_path_CiB_real_entry.delete(0,'end')
    image_path_CiB_imag_entry.delete(0,'end')

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

# 创建文件夹
# record the number of calls of mkdir
ind_mkdir = 1
today_date = datetime.datetime.now().strftime('%Y-%m-%d')
def mkdir(savedata_path_parent):
    global ind_mkdir, savedata_path

    if ind_mkdir == 1:
        savedata_path = os.path.join(savedata_path_parent, today_date,'init')
        folder = os.path.exists(savedata_path)
        if folder:
            print('parent workspace: %s already exists' % savedata_path)
        else:
            os.makedirs(savedata_path)
            print('parent workspace: %s created' % savedata_path)

        f = open(os.path.join(savedata_path, 'plotdata.txt'), 'w')
        f.write('{},{},{},{}\n'.format(0, 1., 1., 1.))
        f.close()

        im = open(os.path.join(savedata_path, 'image_recon.txt'), 'w')
        im.write('{} {}\n{} {}\n'.format(0., 0., 0., 0.))
        im.close()

    else:
        savedata_path = os.path.join(savedata_path_parent, today_date, '1')
        folder = os.path.exists(savedata_path)
        count = 1
        while folder:
            count += 1
            savedata_path = os.path.join(savedata_path_parent, today_date, str(count))
            folder = os.path.exists(savedata_path)
        os.makedirs(savedata_path)  # 创建文件时如果路径不存在会创建这个路径
        print('workspace path: %s created' % savedata_path)
        f = open(os.path.join(savedata_path, 'plotdata.txt'), 'w')
        f.write('{},{},{},{}\n'.format(0, 1., 1., 1.))
        f.close()

        im = open(os.path.join(savedata_path, 'image_recon.txt'), 'w')
        im.write('{} {}\n{} {}\n'.format(0., 0., 0., 0.))
        im.close()
    return savedata_path

def animate_thread(i):
    threading.Thread(target=animate, name='Thread-print', args=(i,)).start()

def run_ptycho():
    threading.Thread(target=start_run, name='Thread-main').start()

############## for test ################
def animate(i):
    file_name = os.path.join(savedata_path, 'plotdata.txt')
    image_name = os.path.join(savedata_path, 'image_recon.txt')
    pullData = open(file_name, "r").read()
    dataList = pullData.split('\n')
    Iter = []
    resi_DR_y = []
    relative_DR_yIMsmall1_5 = []
    relative_DR_maskLPS = []
    for eachLine in dataList:
        if len(eachLine) > 1:
            count_DR, res, rel_LPS_x, rel_LPS_mask = eachLine.split(',')
            Iter.append(int(count_DR))
            resi_DR_y.append(float(res))
            relative_DR_yIMsmall1_5.append(float(rel_LPS_x))
            relative_DR_maskLPS.append(float(rel_LPS_mask))

    pullImage = open(image_name, "r").read()
    pull_list = pullImage.split('\n')
    image = [[float(num) for num in line.split(' ')] for line in pull_list if len(line) > 1]
    # mask_recon.clear()
    # mask_recon.imshow(x__t)
    # mask_recon.colorbar()
    # image_recon.clear()
    # mask_recon.imshow(x__t)
    # mask_recon.colorbar()

    image_recon.clear()
    image_recon.imshow(image,cmap='gray')
    image_recon.set_axis_off()

    error_plot.clear()
    error_plot.set_title('Error Plot', size ='12')
    error_plot.set_xlabel('epoch', fontsize= '10')
    error_plot.set_ylabel('error', fontsize= '10')

    #error_plot.semilogy(Iter[0], resi_DR_y[0], 'ro-')
    #error_plot.semilogy(Iter[0], relative_DR_yIMsmall1_5[0], 'b^:')
    #error_plot.semilogy(Iter[0], relative_DR_maskLPS[0], 'ks--')
    #error_plot.legend(('residual', 'relative Mask', 'relative Image'),
    #           loc='upper right')

    #error_plot.semilogy(Iter[0:-1:5], resi_DR_y[0:-1:5], 'ro')
    #error_plot.semilogy(Iter[0:-1:5], relative_DR_yIMsmall1_5[0:-1:5], 'b^')
    #error_plot.semilogy(Iter[0:-1:5], relative_DR_maskLPS[0:-1:5], 'ks')

    error_plot.semilogy(Iter, resi_DR_y, 'r-')
    error_plot.semilogy(Iter, relative_DR_yIMsmall1_5, 'b:')
    error_plot.semilogy(Iter, relative_DR_maskLPS, 'k--')
    error_plot.legend(('residual', 'relative Mask', 'relative Image'),
                      loc='upper right')


##########################    
    
###### save all parameters to textfile and start running phase retreival 
def start_run():
    global ind_mkdir, savedata_path
    ind_mkdir += 1
    savedata_path_parent = os.path.join(os.getcwd(), 'workspace')
    savedata_path = mkdir(savedata_path_parent)
    today_date=datetime.datetime.now().strftime('%Y-%m-%d')
    #generate savespace
    input_parameters = {'savedata_path': savedata_path}
    #n_h = n_horizontal_entry.get()
    #input_parameters['n_horizontal'] = n_h

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

    MaxIt=MaxIter.get()
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

    salton=salt_on.get()
    input_parameters['salt_on'] = salton

    if salton == 1:
        salt_prob = salt_noise.get()
        input_parameters['salt_prob'] = salt_prob
        salt_inten1 = salt_inten.get()
        input_parameters['salt_inten'] = salt_inten1

    
    with open(os.path.join(savedata_path, 'config.txt'),'w') as f:
        #f.write('n_horizontal=%s \n' % n_h)
        f.write('n_vertical=%s \n' % n_v)
        f.write('overlap_r=%s \n' % OLR)
        f.write('background=%s \n' % bg)
        if bg == 'Fixed':
            f.write('fixed_bd_value = %s \n' % fixed_bd_v)
        f.write('rank = %s \n' % rank)
        f.write('perturb = %s \n' % perturb)

        f.write('mask_type=%s \n' % mask_type)
        f.write('image_type = %s \n' % image_type)
        if image_type =='CiB_image':
            f.write('real_path = {} \nimag_path = {} \n'.format(image_path['real'], image_path['imag']))
        else:
            f.write('image_path = {} \n'.format(image_path))
        
        f.write('mask_delta=%s \n' % mask_dlt)
        f.write('MaxIter=%s \n' % MaxIt)
        f.write('MaxInner=%s\n' % MaxInner)
        f.write('Toler=%s \n' % Tol)
        f.write('os_rate=%s \n' % os_rate)
        f.write('pois_or_gau = %s \n' % pois_or_gau)
        f.write('gamma = %s \n' % ga)
        f.write('salt_on = %s \n' % salton)
        if salton == 1:
            f.write('salt_prob=%s\n' % salt_prob)
            f.write('salt_inten=%s\n'% salt_inten1)
    '''
    input_parameters = {'n_horizontal':10, 'n_vertical': 10, 'overlap_r': 0.5,
                'bg_value': 'Fixed',
                'fixed_bd_value': 20,
                'rank': 'one',
                'perturb': 3,
                'mask_type': 'Correlated',
                'image_type': 'CiB_image',
                'image_path_real': '/Cameraman.png',
                'image_path_imag': '/Barbara256.png',
                'mask_delta': 0.5,
                'MaxIter': 20,
                'MaxInner': 30
                'Toler': 0.00001,
                'os_rate': 2,
                'gamma': 1,
                'salt_on': 0,
                'pois_gau':'poisson',
                'savedata_path': savedata_path,
                'salt_noise': 0.01 }  '''
    ptycho(input_parameters)


        
    
  
#######################################################################    
    
'''Start to construct panel '''
    
    
    

PR_panel = tk.Tk()
PR_panel.title('PhaseReTr_Simulater')

panel_width = 1100
panel_height = 800

screenwidth = PR_panel.winfo_screenwidth()  
screenheight = PR_panel.winfo_screenheight()  
size = '%dx%d+%d+%d' % (panel_width, panel_height, (screenwidth - panel_width)/2, (screenheight - panel_height)/2)
PR_panel.geometry(size) # width*height + pos_x + pos_y


####
#### SCAN TYPE #### 
####
SCAN_TYPE = tk.LabelFrame(PR_panel, width = 96, height =96, text = 'SCAN TYPE')
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
MASKPROP_TYPE = tk.LabelFrame(PR_panel, width = 96, height =96, text = 'MASK PROPERTY')
MASKPROP_TYPE.grid(row=1, column = 1,padx = 6, sticky = 'wens')
#### mask_type

mask_type_value=tk.StringVar()
mask_type_label=tk.Label(MASKPROP_TYPE, text='mask type')
mask_type_label.grid(row=0, column=0, sticky='w')
mask_type = ttk.Combobox(MASKPROP_TYPE,width=15,textvariable=mask_type_value)
mask_type['values']=('Fresnel','Correlated','IID')
mask_type.grid(row=0,column=1,padx=3,pady=3,sticky='w')

####
####   image type
####
IMAGE_TYPE = tk.LabelFrame(PR_panel, width = 160, height =400, text = 'IMAGE TYPE')
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
ALG_SET = tk.LabelFrame(PR_panel, width = 96, height =96, text = 'ALGORITHM PARAMETER')
ALG_SET.grid(row=2, column = 0,padx = 6, sticky = 'wnes')


tk.Label(ALG_SET, text='mask delta').grid(row=0, column=0, sticky='w')
mask_delta=tk.StringVar()
mask_delta_entry = tk.Entry(ALG_SET, textvariable=mask_delta,relief='solid',width=3)
mask_delta_entry.grid(row=0,column=1)

tk.Label(ALG_SET, text='MaxIter').grid(row=1, column=0, sticky='w')
MaxIter=tk.StringVar()
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
gamma_entry = tk.Entry(ALG_SET, textvariable = gamma, relief='solid', width=3)
gamma_entry.grid(row=5, column=1)

# inner loop #
inner_loop_maximum=tk.StringVar()



####
#### NOISE triggered
####

NOISE_TRIGGER = tk.LabelFrame(PR_panel, width = 96, height =96, text = 'Noise Trigger')
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

# start button #
#start_button=tk.Button(PR_panel,text='start', command=start_run, relief='solid')
start_button=tk.Button(PR_panel,text='start', command=run_ptycho, relief='solid')
start_button.grid(row=3, column = 0, sticky='n')
start_button.config(height = 2, width = 7)

# exit button #
exit_button=tk.Button(PR_panel, text="exit", command = PR_panel.quit, relief='solid')
exit_button.grid(row=3, column = 1, sticky='n')
exit_button.config(height = 2, width = 7)
# clear all button #
clear_all_button=tk.Button(PR_panel, text='clear all', relief='solid',command=clear_all)
#CLEAR_ALL_BUTTON.bind('<<Button-1>>', clear_all)
clear_all_button.grid(row=3, column = 2, sticky='n')
clear_all_button.config(height = 2, width = 7)

### matplot live in tkinter

figure_panel = tk.LabelFrame(PR_panel, width = 100, height=50, text = 'figure panel')
figure_panel.grid(row = 2, column = 2, rowspan = 1, sticky = 'wnes')
style.use("ggplot")

f = Figure(figsize=(5, 5), dpi=100)
error_plot = f.add_subplot(211)
image_recon = f.add_subplot(212)
f.subplots_adjust(wspace=0, hspace=0.3)

canvas = FigureCanvasTkAgg(f, figure_panel)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(canvas, figure_panel)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
savedata_path_parent = os.path.join(os.getcwd(), 'workspace')
savedata_path = mkdir(savedata_path_parent)

ani = animation.FuncAnimation(f, animate_thread, interval=15000)


# imshow
#imshow_panel = tk.LabelFrame(PR_panel, width = 100, height=50, text = 'imshow panel')
#imshow_panel.grid(row = 2, column = 3, rowspan = 1, sticky = 'wnes')





### end matplot line in tkinter
PR_panel.mainloop()




















