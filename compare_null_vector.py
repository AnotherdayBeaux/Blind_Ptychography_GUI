from blind_ptychography import *
import numpy as np
import matplotlib.pyplot as plt

from null_vector_fun import *

ini = np.random.uniform(size=(256, 256))# common start point

MaxIter = 1000
rel_im1, patch_1= null_vector_fun(n_vertical = 13, overlap_r = 0.5, tao_ = 0.5, pertb= 0, IterA=MaxIter, IM_ini = ini)

rel_im2, patch_2= null_vector_fun(n_vertical = 13, overlap_r = 0.5, tao_ = 0.5, pertb= 3, IterA=MaxIter, IM_ini = ini)

rel_im3, patch_3= null_vector_fun(n_vertical = 10, overlap_r = 0.5, tao_ = 0.5, pertb= 0, IterA=MaxIter, IM_ini = ini)

rel_im4, patch_4= null_vector_fun(n_vertical = 10, overlap_r = 0.5, tao_ = 0.5, pertb= 3, IterA=MaxIter, IM_ini = ini)

rel_im7, patch_7= null_vector_fun(n_vertical = 10, overlap_r = 0.7, tao_ = 0.5, pertb= 0, IterA=MaxIter, IM_ini = ini)

rel_im8, patch_8= null_vector_fun(n_vertical = 10, overlap_r = 0.7, tao_ = 0.5, pertb= 3, IterA=MaxIter, IM_ini = ini)

rel_im5, patch_5= null_vector_fun(n_vertical = 7, overlap_r = 0.5, tao_ = 0.5, pertb= 0, IterA=MaxIter, IM_ini = ini)

rel_im6, patch_6= null_vector_fun(n_vertical = 7, overlap_r = 0.5, tao_ = 0.5, pertb= 3, IterA=MaxIter, IM_ini = ini)



plt.style.use('ggplot')
plt.plot(np.arange(1, MaxIter+1), rel_im1, 'r', np.arange(1, MaxIter+1), rel_im2, 'k')
plt.plot(np.arange(1, MaxIter+1), rel_im3, 'r:', np.arange(1, MaxIter+1), rel_im3, 'k:')
plt.plot(np.arange(1, MaxIter+1), rel_im5, 'r--', np.arange(1, MaxIter+1), rel_im6, 'k--')
plt.plot(np.arange(1, MaxIter+1), rel_im7, 'r^-', np.arange(1, MaxIter+1), rel_im8, 'k^-')

plt.legend(('pert=0 patch={}'.format(patch_1), 'pert=+-3 patch={}'.format(patch_2), \
            'pert=0 patch={}'.format(patch_3), 'pert=+-3 patch={}'.format(patch_4), \
            'pert=0 patch={}'.format(patch_5), 'pert=+-3 patch={}'.format(patch_6), \
            'pert=0 patch={} olr={}'.format(patch_7, 0.7), 'pert=+-3 patch={} olr={}'.format(patch_8, 0.7), \
            ), loc='upper right')
plt.xlabel('iter', fontsize='10')
plt.ylabel('rel error', fontsize='10')
plt.savefig('plot.png')
plt.show()