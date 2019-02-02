# Blind_Ptychography_GUI
Blind_Ptychography python code supplements to https://arxiv.org/abs/1809.00962


python	--version     3.6
matplotlib    		  3.0.2
tkinter               8.6.8

Starter: GUI_ptycho 

Feb 1st, 2019
When eliminating linear phase shift, one can apply fft2
in high dimension to phase difference f_0*conj(f_k). In 
this code, average phase differences between [0][0], [9][0]
,[0][9] are regarded as linear phase shift, kinda rough 
esimate

In fixed boundary/dark boundary case, boundary is not 
enforced after each epoch. So one should expect linear 
phase shift happens always.


...



