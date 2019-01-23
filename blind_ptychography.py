#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 02:00:39 2018

@author: Zheqing Zhang
email:  zheqing@math.ucdavis.edu


input of function init_Ptycho()
n=n_x, n_y
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import spdiags
    

###### oversampled_ifft & fft with different number of mask 

######## one id mask
def Nos_fft_num_mask0(X, os_rate, mask):
    Na, Nb=X.shape
    
    num_of_masks=1
    Y = np.zeros((os_rate*Na, os_rate*Nb), dtype=complex)  
    for kk in range(os_rate):
        LL_vec = np.exp(-2*kk*1j*np.pi/os_rate * np.linspace(0,(Na-1)/Na, Na))
        LL = np.diag( LL_vec )
        
        for ii in range(os_rate):
            RR_vec = np.exp(-2*ii*1j*np.pi/os_rate * np.linspace(0,(Nb-1)/Nb, Nb))
            RR = np.diag(RR_vec)
            X1 = LL@X@RR
            Y1 = np.fft.fft2(X1)
            Y[kk:os_rate*Na:os_rate][:,ii:os_rate*Nb:os_rate]=Y1
    
    Y = Y*1/np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return Y

####### one random mask

def Nos_fft_num_mask1(X, os_rate, mask):
    Na, Nb = X.shape
    
    num_of_masks = 1
    X = mask*X
    Y = np.zeros((os_rate*Na, os_rate*Nb), dtype=complex)  
    for kk in range(os_rate):
        LL_vec = np.exp(-2*kk*1j*np.pi/os_rate * np.linspace(0, (Na-1)/Na, Na))
        LL = np.diag(LL_vec)
        
        for ii in range(os_rate):
            RR_vec = np.exp(-2*ii*1j*np.pi/os_rate * np.linspace(0, (Nb-1)/Nb, Nb))
            RR = np.diag(RR_vec)
            X1 = LL@X@RR
            Y1 = np.fft.fft2(X1)
            Y[kk:os_rate*Na:os_rate][:,ii:os_rate*Nb:os_rate] = Y1
    
    Y = Y*1/np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return Y
 
    

######## two different random mask
    
def Nos_fft_num_mask2(X, os_rate, mask):
    Na, Nb = X.shape
    
    num_of_masks = 2
    mask1X = mask[0]*X
    mask2X = mask[1]*X
    Y = np.zeros((os_rate*Na, os_rate*Nb*2), dtype=complex)  
    for kk in range(os_rate):
        LL_vec = np.exp(-2*kk*1j*np.pi/os_rate * np.linspace(0,(Na-1)/Na, Na))
        LL = np.diag(LL_vec)
        
        for ii in range(os_rate):
            RR_vec = np.exp(-2*ii*1j*np.pi/os_rate * np.linspace(0,(Nb-1)/Nb, Nb))
            RR = np.diag(RR_vec)
            
            X1 = LL@mask1X@RR
            X2 = LL@mask2X@RR
            Y1 = np.fft.fft2(X1)
            Y2=np.fft.fft2(X2)
            Y[kk:os_rate*Na:os_rate][:, ii:os_rate*Nb:os_rate] = Y1
            Y[kk:os_rate*Na:os_rate][:, os_rate*Nb+ii:os_rate*Nb*2:os_rate] = Y2
    
    Y = Y*1/np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return Y


######### one id mask and one random mask

def Nos_fft_num_mask3(X, os_rate, mask):
    Na, Nb = X.shape
    
    num_of_masks = 3
    mask1X = mask[0]*X
    
    Y = np.zeros((os_rate*Na, os_rate*Nb*2), dtype=complex)  
    for kk in range(os_rate):
        LL_vec = np.exp(-2*kk*1j*np.pi/os_rate * np.linspace(0, (Na-1)/Na,Na))
        LL = np.diag(LL_vec)
        
        for ii in range(os_rate):
            RR_vec = np.exp(-2*ii*1j*np.pi/os_rate * np.linspace(0, (Nb-1)/Nb,Nb))
            RR = np.diag( RR_vec )
            
            X1 = LL@mask1X@RR
            X2 = LL@X@RR
            Y1 = np.fft.fft2(X1)
            Y2 = np.fft.fft2(X2)
            Y[kk:os_rate*Na:os_rate][:, ii:os_rate*Nb:os_rate] = Y1
            Y[kk:os_rate*Na:os_rate][:, os_rate*Nb+ii:os_rate*Nb*2:os_rate] = Y2
    
    Y = Y*1/np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return Y
    


#######   ifft part  
######   one id mask
def Nos_ifft_num_mask0(Y, os_rate, mask):   
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na/os_rate)
    Nb = int(Y_Nb/os_rate)
    num_of_masks=1 
    X=np.zeros((Na, Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2*ii*1j*np.pi/os_rate * np.linspace(0, (Na-1)/Na, Na))
        LL= np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2*kk*1j*np.pi/os_rate * np.linspace(0, (Nb-1)/Nb, Nb))
            RR = np.diag(RR_vec)
            
            sub_Y = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:Y_Nb:os_rate])
            X1 = LL@sub_Y@RR
            X = X+X1
            
    X = X*np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return X


###### one random mask
def Nos_ifft_num_mask1(Y, os_rate, mask):   
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na/os_rate)
    Nb = int(Y_Nb/os_rate)
    num_of_masks = 1
    X = np.zeros((Na,Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2*ii*1j*np.pi/os_rate * np.linspace(0,(Na-1)/Na, Na))
        LL = np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2*kk*1j*np.pi/os_rate * np.linspace(0,(Nb-1)/Nb, Nb))
            RR = np.diag(RR_vec)
            
            sub_Y = np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:Y_Nb:os_rate])
            X1 = LL@sub_Y@RR
            X = X+X1
    X = mask.conj()*X
    X = X*np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return X  




###### two random mask
def Nos_ifft_num_mask2(Y, os_rate, mask):
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na/os_rate)
    Nb = int(Y_Nb/os_rate/2)
    num_of_masks = 2
    X2 = np.zeros((Na,Nb), dtype=complex)
    X = np.zeros((Na,Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2*ii*1j*np.pi/os_rate * np.linspace(0,(Na-1)/Na,Na))
        LL= np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2*kk*1j*np.pi/os_rate * np.linspace(0,(Nb-1)/Nb,Nb))
            RR=np.diag(RR_vec)
            
            sub_Y1=np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:int(Y_Nb/2):os_rate])
            X1=LL@sub_Y1@RR
            X=X+X1
            
            sub_Y2=np.fft.ifft2(Y[ii:Y_Na:os_rate][:, int(Y_Nb/2)+kk:Y_Nb:os_rate])
            X22=LL@sub_Y2@RR
            X2=X2+X22
    X=mask[0].conj()*X
    X2=mask[1].conj()*X2
    X=X+X2
    X=X*np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return X    



###### 1.5 mask
def Nos_ifft_num_mask3(Y, os_rate, mask):   
    Y_Na, Y_Nb = Y.shape
    Na = int(Y_Na/os_rate)
    Nb = int(Y_Nb/os_rate)
    num_of_masks=2 
    X2=np.zeros((Na,Nb), dtype=complex)
    X=np.zeros((Na,Nb), dtype=complex)
    for ii in range(os_rate):
        LL_vec = np.exp(2*ii*1j*np.pi/os_rate * np.linspace(0,(Na-1)/Na,Na))
        LL= np.diag(LL_vec)
        for kk in range(os_rate):
            RR_vec = np.exp(2*kk*1j*np.pi/os_rate * np.linspace(0,(Nb-1)/Nb,Nb))
            RR=np.diag(RR_vec)
            
            sub_Y1=np.fft.ifft2(Y[ii:Y_Na:os_rate][:, kk:int(Y_Nb/2):os_rate])
            X1=LL@sub_Y1@RR
            X=X+X1
            
            sub_Y2=np.fft.ifft2(Y[ii:Y_Na:os_rate][:, int(Y_Nb/2)+kk:Y_Nb:os_rate])
            X22=LL@sub_Y2@RR
            X2=X2+X22
    X = mask[0].conj()*X
    X = X+X2
    X = X*np.sqrt(Na*Nb)/np.sqrt(num_of_masks*os_rate**2)
    return X 

###### 3 mask case
######## to be continued.
fft_dict={'id mask':Nos_fft_num_mask0, 'one mask':Nos_fft_num_mask1, 'two mask': Nos_fft_num_mask2, '1.5 mask': Nos_fft_num_mask3}
ifft_dict={'id mask':Nos_ifft_num_mask0, 'one mask':Nos_ifft_num_mask1, 'two mask': Nos_ifft_num_mask2, '1.5 mask': Nos_ifft_num_mask3}

    

    
    
    
    
    
    
    
#### in ptychography the num_mask is predetermined to be 1. 
### image space fft/ifft
# fft periodic boundary case

def ptycho_Im_PeriB_fft(X, os_rate, mask, l_patch, x_c_p, y_c_p, BackGd):
    half_patch = int((l_patch-1)/2)
    Na, Nb = X.shape
    Big_X = np.kron(np.ones((3,3), dtype=complex), X)

    subNa, subNb = x_c_p.shape
    Y=np.zeros((os_rate*l_patch*subNa,os_rate*l_patch*subNb), dtype=complex)
    for i in range(subNb):        
        for j in range(subNa):
            center_x = Na+x_c_p[j][i]
            center_y = Nb+y_c_p[j][i]
            piece = Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch:center_y+half_patch+1]
            piece_Y = Nos_fft_num_mask1(piece,os_rate,mask)
            Y[j*os_rate*l_patch : (j+1)*os_rate*l_patch][:, i*os_rate*l_patch : (i+1)*os_rate*l_patch] = piece_Y

    return Y

   
# fft fixed boundary case     
def ptycho_Im_FixedB_fft(X, os_rate, mask, l_patch, x_c_p, y_c_p, BackGd):
    half_patch = int((l_patch-1)/2)
    Na, Nb = X.shape
    Big_X=BackGd    # BackGd, if want to enforce the intensity, use BackGd = intensity
    Big_X[Na:2*Na][:, Nb:2*Nb] = X
    subNa, subNb= x_c_p.shape
    Y=np.zeros((os_rate*l_patch*subNa,os_rate*l_patch*subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x = Na+x_c_p[j][i]
            center_y = Nb+y_c_p[j][i]
            piece=Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch:center_y+half_patch+1]
            piece_Y=Nos_fft_num_mask1(piece,os_rate,mask)
            Y[j*os_rate*l_patch : (j+1)*os_rate*l_patch][:, i*os_rate*l_patch : (i+1)*os_rate*l_patch] = piece_Y

    return Y
    
# ifft periodic boundary case   
def ptycho_Im_PeriB_ifft(Y, Na, Nb, os_rate, mask, l_patch, x_c_p, y_c_p):
    half_patch = int((l_patch-1)/2)
    Big_X = np.zeros((3*Na,3*Nb), dtype=complex)
    
    subNa, subNb = x_c_p.shape
    
    
    for i in range(subNa):
        for j in range(subNb):
            center_x = x_c_p[i][j]+Na
            center_y = y_c_p[i][j]+Nb
            piece_Y = Y[i*os_rate*l_patch : (i+1)*os_rate*l_patch][:,j*os_rate*l_patch : (j+1)*os_rate*l_patch]
            piece_X = Nos_ifft_num_mask1(piece_Y,os_rate,mask)
            Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch: center_y+half_patch+1] = \
                Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch: center_y+half_patch+1]+piece_X
    X = np.zeros((Na,Nb), dtype=complex)
    for i in range(3):
        for j in range(3):
            X = Big_X[i*Na:(i+1)*Na][:,j*Nb:(j+1)*Nb]+X
    return X, Big_X        



# ifft fixed boundary case     
def ptycho_Im_FixedB_ifft(Y, Na, Nb, os_rate, mask, l_patch, x_c_p, y_c_p):
    half_patch = int((l_patch-1)/2)
    Big_X = np.zeros((3*Na,3*Nb), dtype=complex)
    
    subNa, subNb=x_c_p.shape

    for i in range(subNa):
        for j in range(subNb):
            center_x= x_c_p[i][j]+Na
            center_y=y_c_p[i][j]+Nb
            piece_Y=Y[i*os_rate*l_patch: (i+1)*os_rate*l_patch][:, j*os_rate*l_patch: (j+1)*os_rate*l_patch]
            piece_X=Nos_ifft_num_mask1(piece_Y,os_rate,mask)
            Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch: center_y+half_patch+1]= \
                Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch: center_y+half_patch+1]+piece_X
    X=Big_X[Na:2*Na][:,Nb:2*Nb]
    return X, Big_X  

### mask space fft/ifft
def pr_phase_perib_fft(phase, os_rate, X, l_patch, x_c_p, y_c_p, BackGd):
    
    Na, Nb = X.shape
    Big_X = np.kron(np.ones((3,3), dtype=complex),X)    ### enforce the periodic boundary condition will improve performance
                                       ### while in fixed boundary case we don't
    half_patch = int((l_patch-1)/2)
    subNa, subNb = x_c_p.shape
    fft_phase=np.zeros((os_rate*l_patch*subNa,os_rate*l_patch*subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x=x_c_p[j][i]+Na
            center_y=y_c_p[j][i]+Nb
            mask=Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch: center_y+half_patch+1]
            fft_patch_phase=Nos_fft_num_mask1(phase,os_rate,mask)
            fft_phase[os_rate*l_patch*j: os_rate*l_patch*(j+1)][:,os_rate*l_patch*i: os_rate*l_patch*(i+1)]=fft_patch_phase
    
    return fft_phase
    
####    
def pr_phase_perib_ifft(fft_phase, os_rate,X, l_patch, x_c_p, y_c_p, BackGd):
    Na,Nb=X.shape
    Big_X= np.kron(np.ones((3,3), dtype=complex), X)
    half_patch = int((l_patch-1)/2)
    subNa,subNb=x_c_p.shape
    phase=np.zeros((l_patch, l_patch), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x=x_c_p[j][i]+Na
            center_y=y_c_p[j][i]+Nb
            
            mask=Big_X[center_x-half_patch:center_x+half_patch+1][:, center_y-half_patch: center_y+half_patch+1]
            fft_patch_phase=fft_phase[j*l_patch*os_rate:(j+1)*l_patch*os_rate][:,i*l_patch*os_rate:(i+1)*l_patch*os_rate]
            ifft_patch_phase=Nos_ifft_num_mask1(fft_patch_phase,os_rate,mask)
            phase=phase+ifft_patch_phase
                 
    return phase

    
    
def pr_phase_fixedb_fft(phase,os_rate,X,l_patch,x_c_p,y_c_p, BackGd):
    Na, Nb = X.shape
    Big_X =BackGd
    Big_X[Na:2*Na][:,Nb:2*Nb]=X
    half_patch = int((l_patch-1)/2)
    subNa, subNb = x_c_p.shape
    fft_phase=np.zeros((os_rate*l_patch*subNa,os_rate*l_patch*subNb), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x=x_c_p[j][i]+Na
            center_y=y_c_p[j][i]+Nb
            mask=Big_X[center_x-half_patch:center_x+half_patch+1][:,center_y-half_patch: center_y+half_patch+1]
            fft_patch_phase=Nos_fft_num_mask1(phase,os_rate,mask)
            fft_phase[os_rate*l_patch*j: os_rate*l_patch*(j+1)][:,os_rate*l_patch*i: os_rate*l_patch*(i+1)]=fft_patch_phase
    
    return fft_phase
    
    
def pr_phase_fixedb_ifft(fft_phase,os_rate,X,l_patch,x_c_p,y_c_p,BackGd):
    Na,Nb=X.shape
    Big_X= BackGd
    Big_X[Na:2*Na][:,Nb:2*Nb]=X
    half_patch = int((l_patch-1)/2)
    subNa,subNb=x_c_p.shape
    phase=np.zeros((l_patch, l_patch), dtype=complex)
    for i in range(subNb):
        for j in range(subNa):
            center_x=x_c_p[j][i]+Na
            center_y=y_c_p[j][i]+Nb
            
            mask=Big_X[center_x-half_patch:center_x+half_patch+1][:, center_y-half_patch: center_y+half_patch+1]
            fft_patch_phase=fft_phase[j*l_patch*os_rate:(j+1)*l_patch*os_rate][:,i*l_patch*os_rate:(i+1)*l_patch*os_rate]
            ifft_patch_phase=Nos_ifft_num_mask1(fft_patch_phase,os_rate,mask)
            phase=phase+ifft_patch_phase
                 
    return phase


#### poisson likelihood function Z is the  data we collected ~ fft(mask*X)**2   
def poisson_likely(Q,Z,gamma):
    rot_z= (Q/gamma+np.sqrt(Q**2/gamma**2+8*(2+1/gamma)*Z))/(4+2/gamma)
    return rot_z
        
def gaussian_likely(Q,Z,gamma):
    rot_z=(np.sqrt(Z)+1/gamma*Q)/(1+1/gamma)
    return rot_z

# generate corelated mask's 
# m = mask size; c_l= corelation distance; convolve iid phase_arg
def Cor_Mask(m, c_l):
    BigMask_phase=np.random.uniform(size = (m+c_l,m+c_l))

    BigMask=np.exp((BigMask_phase-0.5)*np.pi*2j)

    Cor_mask=np.zeros((m, m), dtype=complex)
    for i in range(c_l):
        for j in range(c_l):
            Cor_mask = Cor_mask+BigMask[i:m+i][:,j:m+j]

    Cor_mask= Cor_mask/np.abs(Cor_mask)
    Cor_mask= np.angle(Cor_mask)/2/np.pi
    return Cor_mask








    
    
def getrid_LPS_m(f_0, f_k, n):
    Na, Nb = n[0], n[1]
    
    exp_2pikn = f_0*f_k.conj()
    calib_exp = exp_2pikn*exp_2pikn[0][0].conj()
    angle_exp = np.angle(calib_exp)
    
    rec_l1, rec_k1 = -angle_exp[0][9]/9/2/np.pi*Nb, -angle_exp[9][0]/9/2/np.pi*Na

    return rec_k1, rec_l1   
    
def Proj_on_Image(Na,Nb, os_rate,lambda_t, mask_estimate, l_patch, x_c_p, y_c_p, DR_update_fun ):

    x_t, Big_x_t = ptycho_Im_PeriB_ifft(lambda_t,Na,Nb,os_rate,mask_estimate,l_patch,x_c_p,y_c_p)
    x__t=x_t/nor_ptycho
    Big_x__t=Big_x_t/Big_nor
    AAlambda=ptycho_Im_PeriB_fft(x__t,os_rate,mask,mask_estimate,l_patch,x_c_p,y_c_p,Big_x__t)
    y_tplus1=AAlambda
    
    ylambda_k=2*y_tplus1-lambda_t
    Q=np.abs(ylambda_k)
    
    rot_z=poisson_likely(Q,Z,gamma)
    
    z_tplus1=rot_z*ylambda_k/Q
    lambda_t=lambda_t+z_tplus1-y_tplus1
    
    return lambda_t, x__t, Big_x__t
    
    
    
    
    
    
    
def Image_update(Na,Nb, os_rate,lambda_t, mask_estimate, l_patch, x_c_p, y_c_p, DR_update_fun ):
    MaxIter_u_ip=30
    update_im=1
    #Na, Nb= IM.shape
    IM_=np.ones((Na, Nb),dtype = 'float')
    p_fft_IM=ptycho_Im_PeriB_fft(IM_,os_rate,mask_estimate,l_patch,x_c_p,y_c_p)
    nor_ptycho, Big_nor = ptycho_Im_PeriB_ifft(p_fft_IM,Na,Nb,os_rate,mask_estimate,l_patch,x_c_p,y_c_p)
    residual_x=np.array([0,0,0])
    resi_diff_im=1
    
    while (update_im < 5  or ( update_im < MaxIter_u_ip  and resi_diff > 1e-5)):
        
        lambda_t, x__t, Big_x__t = Proj_on_Image(Na,Nb, os_rate,lambda_t, mask_estimate, l_patch, x_c_p, y_c_p, DR_update_fun )
        
        
    
    
        # calculate the termination error
        #ee=np.abs(IM.reshape(-1).conj().dot(x__t.reshape(-1)))/IM.reshape(-1).conj().dot(x__t.reshape(-1))
        #rel_xx = np.linalg.norm(ee*x__t-IM, 'fro')/np.linalg.norm(IM,'fro')
        
        res_x = np.linalg.norm(np.abs(Y)-np.abs(y_tplus1),'fro')/norm_Y
        
        residual_x[update_im+2]= res_x
        resi_diff_im=1/3*np.linalg.norm(residual_x[update_im:update_im+3]-residual_x[update_im-1:update_im+2],1)/res_x
        update_im=update_im+1
        
    return  lambda_t, x__t, Big_x__t
    
    
    
    
def Proj_on_Mask(self, x__t):
    nor_mask_tplus1=pr_phase_perib_ifft(fft_phase,os_rate,x__t,l_patch,x_c_p,y_c_p)/nor_phase
    
    mask_tplus1=nor_mask_tplus1/np.abs(nor_mask_tplus1)  # enforce unimodular constraint
    fft_phase_1=pr_phase_perib_fft(mask_tplus1,os_rate,x__t,l_patch,x_c_p,y_c_p)
    Q_phase=2*fft_phase_1-fft_phase
    Q=np.abs(Q_phase)
    rot_z = poisson_likely(Q,Z,gamma)
    z_mask_tplus1 = rot_z*Q_phase/Q
    fft_phase=fft_phase+z_mask_tplus1-fft_phase_1
    
    
def Mask_update(l_patch, os_rate, x__t, x_c_p, y_c_p):
    MaxIter_u_ip = 30 ## self.
    update_phase=1
    ### normalize factor
    phase_=np.ones((l_patch,l_patch))
    fft_phase_=pr_phase_perib_fft(phase_,os_rate,x__t,l_patch,x_c_p,y_c_p)
    nor_phase=pr_phase_perib_ifft(fft_phase_,os_rate,x__t,l_patch,x_c_p,y_c_p)
    
    
    fft_phase=y_tplus1  ##
    residual_xx=np.array([0,0,0])
    resi_diff_phase=1
    
    while  (update_phase < 5 or ( update_phase < MaxIter_u_ip and resi_diff_phase > 1e-5)):
        
        Proj_on_Mask(self)
        
        
        # calculate the termination error
        res_mask=np.linalg.norm(np.abs(fft_phase_1)-b,'fro')/norm_Y  # b =sqrt(Z)
        residual_xx[update_phase+2]= res_mask
        resi_diff_phase=1/3* np.linalg.norm(residual_xx[update_phase:update_phase+3]-residual_xx[update_phase-1:update_phase+2],1)/res_mask
        update_phase= update_phase+1
  







def plot():
    
    
    plt.figure(0)
    plt.title('figure')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    
    plt.semilogy(Iter[0],resi_DR_y[0],'ro-')
    plt.semilogy(Iter[0],relative_DR_yIMsmall1_5[0], 'b^:')
    plt.semilogy(Iter[0],relative_DR_maskLPS[0],'r-')
    plt.legend(('residual', 'relative Mask', 'relative Image'),
           loc='upper right')
    
    plt.semilogy(Iter[0:5:-1],resi_DR_y[0:5:-1],'ro-')
    plt.semilogy(Iter[0:5:-1],relative_DR_yIMsmall1_5[0:5:-1], 'ro')
    plt.semilogy(Iter[0:5:-1],relative_DR_maskLPS[0:5:-1],'r-')
    
    plt.semilogy(Iter,resi_DR_y,'ro-')
    plt.semilogy(Iter,relative_DR_yIMsmall1_5, 'ro')
    plt.semilogy(Iter,relative_DR_maskLPS,'r-')
    
    plt.show()
    plt.savefig('')
    
    #
    plt.figure(1)
    plt.imshow(x__t)
    plt.colorbar()
    plt.show()
    plt.savefig('')
    
    #
    plt.figure(2)
    plt.imshow(np.angle(mask_estimate))
    plt.colorbar()
    plt.show()
    plt.savefig('')
    
# 创建文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # 创建文件时如果路径不存在会创建这个路径
        print('workspace path: %s created' % path)
    else:
        print('path: %s already exists' % path)
        

# 
def ini_ptycho(pass_parameters):
    #pass_parameters = kwargs
    str1 = pass_parameters['str1']
    str2 = pass_parameters['str2']
    os_rate = pass_parameters['os_rate']
    type_im = pass_parameters['type_im']
    l_patch_x = pass_parameters['l_patch_x']
    l_patch_y = pass_parameters['l_patch_y']
    # temp solution
    l_patch = l_patch_x
    l_patch_y = l_patch_x
    #
    x_c_p = pass_parameters['x_c_p']
    y_c_p = pass_parameters['y_c_p']
    bd_con = pass_parameters['bd_con']
    mask_type = pass_parameters['mask_type']
    rho = pass_parameters['rho']
    beta_1 = pass_parameters['beta_1']
    beta_2 = pass_parameters['beta_2']
    c_l = pass_parameters['c_l']
    salt_on = pass_parameters['salt_on']
    SUBIM = pass_parameters['SUBIM']

    ptycho_fft_dict = {'ptycho_Im_PeriB_fft': ptycho_Im_PeriB_fft,
                       'ptycho_Im_FixedB_fft': ptycho_Im_FixedB_fft,
                       'ptycho_Im_PeriB_ifft': ptycho_Im_PeriB_ifft,
                       'ptycho_Im_FixedB_ifft': ptycho_Im_FixedB_ifft,
                       'pr_phase_perib_fft': pr_phase_perib_fft,
                       'pr_phase_fixedb_fft': pr_phase_fixedb_fft,
                       'pr_phase_perib_ifft': pr_phase_perib_ifft,
                       'pr_phase_fixedb_ifft': pr_phase_fixedb_ifft}

    if pass_parameters['bd_con'] == 'Fixed':
        BackGd = pass_parameters['BackGd']

        ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
        ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
        pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
        pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

    elif pass_parameters['bd_con'] == 'Dark':
        BackGd = pass_parameters['BackGd']

        ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_FixedB_fft']
        ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_FixedB_ifft']
        pr_phase_fft = ptycho_fft_dict['pr_phase_fixedb_fft']
        pr_phase_ifft = ptycho_fft_dict['pr_phase_fixedb_ifft']

    else:
        BackGd = pass_parameters['BackGd']
        ptycho_IM_fft = ptycho_fft_dict['ptycho_Im_PeriB_fft']
        ptycho_IM_ifft = ptycho_fft_dict['ptycho_Im_PeriB_ifft']
        pr_phase_fft = ptycho_fft_dict['pr_phase_perib_fft']
        pr_phase_ifft = ptycho_fft_dict['pr_phase_perib_ifft']

    
    # temp use, need modify
    IM1 = plt.imread(str1)
    if len(IM1.shape) >2:   # generated a grey image 
        IM1=IM1.sum(axis=2)
        
    IM2 = plt.imread(str2)
    if len(IM2.shape) >2:
        IM2=IM2.sum(axis=2)

    #IM1 = IM1[0:SUBIM][:,0:SUBIM]   # test for subimage
    #IM1 = IM1.astype(np.float)
    
    #IM2 = IM2[0:SUBIM][:,0:SUBIM]
    #IM2 = IM2.astype(np.float)     # test for subimage
    
    if type_im == 0:
        IM=IM1
    elif type_im == 2:
        IM= IM1+ 1j*IM2
    else:
        IM = IM1* np.exp(2j*np.pi * np.random.uniform(size=IM1.shape))
        
    if salt_on == 1:
        salt_prob = pass_parameters['salt_prob']
        salt_inten = pass_parameters['salt_inten']
        salt_Noise = np.random.uniform(size=IM.shape)
        salt_Noise[salt_Noise < (1-salt_prob)] = 0
        salt_Noise[salt_Noise >=(1-salt_prob)] = salt_inten
        IM = IM+salt_Noise   # need modify 
    
    
    if mask_type == 0: # iid mask
        phase_arg = np.random.uniform(size = (l_patch_x, l_patch_y))
        mask = np.exp(2j * np.pi * phase_arg)
        
    elif mask_type ==1: # correlated mask
        phase_arg = Cor_Mask(l_patch_x,int(l_patch_x/2) ) # please modify the second entry half distance correlation by default
        mask =np.exp(2j * np.pi * phase_arg)
    else: # frensel mask
        phase_arg = 1/2 * rho *(( 1/l_patch_x * ((np.arange(l_patch_x, dtype = float)).reshape(l_patch_x,1)-beta_1)**2) @ np.ones((1,l_patch_y),dtype=float)+ \
                    1/l_patch_y * (np.ones((l_patch_x,1), dtype = float) @ (np.arange(l_patch_y,dtype=float).reshape(1,l_patch_y)-beta_2)**2))
        mask = np.exp(2j * np.pi * phase_arg)
        
    
    # pois or gaus need to be added

    #if bd_con == 'Fixed':
    Y = ptycho_IM_fft(IM, os_rate, mask, l_patch_x, x_c_p, y_c_p, BackGd)

    #elif bd_con == 'Dark':
    #Y = ptycho_Im_FixedB_fft(IM, os_rate, mask, l_patch_x, x_c_p, y_c_p, BackGd)

    #else:  # periodic
    #Y = ptycho_Im_PeriB_fft(IM, os_rate, mask, l_patch_x, x_c_p, y_c_p)
    Z = np.abs(Y)**2

    ### need to add salt_noise
    return IM, Z, phase_arg
    

        
# main program  
# input_parameters from GUI
def ptycho(input_parameters):


    #today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # the dict of likelihood function
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
    b = np.sqrt(Z)
    norm_Y= np.linalg.norm(b,'fro')
    
    # add image_xy_grid and mask_xy_grid

    im_x_grid = np.arange(Na).reshape(Na,1) * np.ones((1,Nb),dtype='float')
    im_y_grid = np.ones((Na,1),dtype='float') * np.arange(Nb).reshape(1,Nb)
    mask_x_grid = np.arange(l_patch_x).reshape(l_patch_x,1) * np.ones((1,l_patch_x),dtype='float')
    mask_y_grid = np.ones((l_patch_x,1),dtype='float') * np.arange(l_patch_x).reshape(1,l_patch_x)
    #

    # l_patch_x l_patch_y need fix 
    mask_estimate = np.exp(2j*np.pi*phase_arg) * \
                    np.exp(mask_delta*2j*np.pi*(np.random.uniform(size=(l_patch_x,l_patch_x))-1/2))
    
    # initial gauss on ptycho_fft(IM)
    lambda_t =np.random.uniform(size=Z.shape)
    
    # true mask
    mask = np.exp(2j*np.pi*phase_arg)
    
    # creat workspace path
    #f=open(os.path.join(savedata_path, 'plotdata.txt'),'w')

    count_DR = 1

    resi_DR_y=np.zeros((MaxIter,), dtype='float')
    relative_DR_yIMsmall1_5=np.zeros((MaxIter,), dtype='float')
    relative_DR_maskLPS=np.zeros((MaxIter,), dtype='float')
    
    res_x = 1.0
    while res_x > Tol and count_DR < MaxIter:
        # image_update
        #MaxIter_u_ip=30
        update_im=1
        
        # calculate nor_factors depenent on mask_estimate
        IM_=np.ones((Na, Nb), dtype='complex')
        p_fft_IM=ptycho_IM_fft(IM_, os_rate, mask_estimate, l_patch, x_c_p, y_c_p, np.ones(BackGd.shape, dtype=complex))
        nor_ptycho, Big_nor = ptycho_IM_ifft(p_fft_IM, Na, Nb, os_rate, mask_estimate, l_patch, x_c_p, y_c_p)
        Big_nor[Big_nor==0]=1.0
        residual_x=np.zeros((MaxIter_u_ip+3,), dtype='float')
        resi_diff_im=1.0
        
        while (update_im < 5  or ( update_im < MaxIter_u_ip  and resi_diff_im > 1e-5)):
            x_t, Big_x_t = ptycho_IM_ifft(lambda_t, Na,Nb,os_rate,mask_estimate,l_patch,x_c_p,y_c_p)
            x__t=x_t/nor_ptycho
            Big_x__t=Big_x_t/Big_nor
            AAlambda = ptycho_IM_fft(x__t, os_rate, mask_estimate, l_patch, x_c_p, y_c_p, Big_x__t)
               #AAlambda=ptycho_Im_PeriB_fft(x__t,os_rate,mask_estimate,l_patch,x_c_p,y_c_p,Big_x__t)
               # Big_x__t: no enforce of prior info
            y_tplus1 = AAlambda
            
            ylambda_k = 2*y_tplus1-lambda_t
            Q = np.abs(ylambda_k)
            
            rot_z = DR_update_fun(Q, Z, gamma)
            
            z_tplus1 = rot_z*ylambda_k/Q
            lambda_t = lambda_t+z_tplus1-y_tplus1
            
        
        
            # calculate the termination error
            #ee=np.abs(IM.reshape(-1).conj().dot(x__t.reshape(-1)))/IM.reshape(-1).conj().dot(x__t.reshape(-1))
            #rel_xx = np.linalg.norm(ee*x__t-IM, 'fro')/np.linalg.norm(IM,'fro')
            
            res_x = np.linalg.norm(np.abs(b)-np.abs(y_tplus1),'fro')/norm_Y
            
            residual_x[update_im+2]= res_x
            resi_diff_im=1/3*np.linalg.norm(residual_x[update_im:update_im+3]-residual_x[update_im-1:update_im+2],1)/res_x
            update_im=update_im+1
            
        # end image update
        
        # mask update
        update_phase=1
        
        # # calculate nor_factors depenent on image
        phase_=np.ones((l_patch,l_patch),dtype='complex')
        fft_phase_=pr_phase_fft(phase_, os_rate, x__t, l_patch, x_c_p, y_c_p, Big_x__t)
        nor_phase=pr_phase_ifft(fft_phase_, os_rate, x__t, l_patch, x_c_p, y_c_p, Big_x__t)
        
        
        fft_phase = y_tplus1  ##
        residual_xx=np.zeros((MaxIter_u_ip+3,), dtype='float')
        resi_diff_phase=1.0
        
        while  (update_phase < 5 or ( update_phase < MaxIter_u_ip and resi_diff_phase > 1e-5)):
            
            nor_mask_tplus1 = pr_phase_ifft(fft_phase, os_rate, x__t, l_patch, x_c_p, y_c_p, Big_x__t)/nor_phase
    
            mask_tplus1=nor_mask_tplus1/np.abs(nor_mask_tplus1)  # enforce unimodular constraint
            fft_phase_1=pr_phase_fft(mask_tplus1, os_rate, x__t, l_patch, x_c_p, y_c_p, Big_x__t)
            Q_phase = 2*fft_phase_1-fft_phase
            Q=np.abs(Q_phase)
            rot_z = poisson_likely(Q,Z,gamma)
            z_mask_tplus1 = rot_z*Q_phase/Q
            fft_phase=fft_phase+z_mask_tplus1-fft_phase_1
            
            
            # calculate the termination error
            res_mask=np.linalg.norm(np.abs(fft_phase_1)-b, 'fro')/norm_Y  # b =sqrt(Z)
            residual_xx[update_phase+2] = res_mask
            resi_diff_phase = 1/3 * np.linalg.norm(residual_xx[update_phase:update_phase+3]-residual_xx[update_phase-1:update_phase+2],1)/res_mask
            update_phase= update_phase+1
        # end mask update   

        mask_estimate = mask_tplus1
        lambda_t = fft_phase_1

        # calculate rel_LPS_mask
        ee_mask = np.abs(mask.reshape(-1).conj().dot(mask_estimate.reshape(-1))) / mask.reshape(-1).conj().dot(mask_estimate.reshape(-1))
        recm_k1, recm_l1 = getrid_LPS_m(mask, ee_mask * mask_estimate, IM.shape)
        mask_t_LPS = np.exp(-2j * np.pi * (mask_x_grid * recm_k1 / Na + mask_y_grid * recm_l1 / Nb)) * mask_estimate
        ee_mask = np.abs(mask.reshape(-1).conj().dot(mask_t_LPS.reshape(-1))) / mask.reshape(-1).conj().dot(mask_t_LPS.reshape(-1))
        rel_LPS_mask = np.linalg.norm(ee_mask * mask_t_LPS - mask, 'fro') / np.linalg.norm(mask, 'fro')

        # calculate rel_LPS_x
        ee_p = np.abs(IM.reshape(-1).conj().dot(x__t.reshape(-1)))/IM.reshape(-1).conj().dot(x__t.reshape(-1))
        rec_k1,rec_l1 = getrid_LPS_m(IM, ee_p*x__t, IM.shape)   # n subim dimension. ###
        #rec_k1, rec_l1 = -recm_k1, -recm_l1
        LPS_x_t=np.exp(-2j*np.pi*(im_x_grid*rec_k1/Na+im_y_grid*rec_l1/Nb))*x__t
        ee=np.abs(IM.reshape(-1).conj().dot(LPS_x_t.reshape(-1)))/(IM.reshape(-1).conj().dot(LPS_x_t.reshape(-1)))
        rel_LPS_x=np.linalg.norm(LPS_x_t*ee-IM,'fro')/np.linalg.norm(IM,'fro')
        #rel_x=np.linalg.norm(ee_p*x__t-IM,'fro')/np.linalg.norm(IM,'fro')
        

        # store data
        resi_DR_y[count_DR-1] = res_x
        relative_DR_yIMsmall1_5[count_DR-1] =rel_LPS_x
        # relative_DR_xIMsmall1_5[count_DR-1] =rel_x
        relative_DR_maskLPS[count_DR-1] = rel_LPS_mask

        with open(os.path.join(savedata_path, 'plotdata.txt'),'a') as f:
            f.write('{},{},{},{}\n'.format(count_DR, res_x, rel_LPS_x, rel_LPS_mask))

        recon_x_t = LPS_x_t * ee
        if image_type == 'real':
            image_recon = np.real(recon_x_t)
        elif image_type == 'rand_phase':
            image_recon = np.abs(recon_x_t)
        else:
            image_recon = np.real(recon_x_t)

        mat = np.matrix(image_recon)
        image_file_name = os.path.join(savedata_path, 'image_recon.txt')
        with open(image_file_name, 'wb') as im:
            for line in mat:
                np.savetxt(im, line, fmt='%f')

        #print('count_DR=%d update_im=%d update_phase=%d \n rel_LPS_x=%s rel_LPS_mask=%s res_x=%s \n' % (
        #count_DR, update_im, update_phase, '{:.4e}'.format(rel_LPS_x), '{:.4e}'.format(rel_LPS_mask),
        #'{:.4e}'.format(res_x)))

        print('count_DR=%d rec_k1=%s rec_l1=%s \n rel_LPS_x=%s rel_LPS_mask=%s res_x=%s \n' % (
        count_DR, '{:.4e}'.format(rec_k1), '{:.4e}'.format(rec_l1), '{:.4e}'.format(rel_LPS_x), '{:.4e}'.format(rel_LPS_mask),
        '{:.4e}'.format(res_x)))

        count_DR += 1

    #f.close()


    #recon_x_t=LPS_x_t*ee
    #if image_type == 'real':
    #    image_recon = np.real(recon_x_t)
    #elif image_type =='rand_phase':
    #    image_recon = np.abs(recon_x_t)
    #else:
    #    image_recon = np.real(recon_x_t)


    #mat = np.matrix(image_recon)
    #image_file_name = os.path.join(savedata_path, 'image_recon.txt')
    #with open(image_file_name, 'wb') as im:
    #    for line in mat:
    #        np.savetxt(im, line, fmt='%f')
        
        
    
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

