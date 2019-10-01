# compare APR/AAR/RAAR in phase retrieval

from blind_ptycho_fun import *


class RAAR_APR_AAR_phase_retrieval(object):
    def __init__(self, input_parameters):
        self.pois_or_gau = input_parameters['pois_gau']
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
        self.Tol = float(input_parameters['Toler'])
        self.rho = float(input_parameters['rho'])
        self.os_rate = int(input_parameters['os_rate'])
        # make sure the im1 and im2 are of the same size in CiB case
        print(self.str1)
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



# generate image based on type of IM
        if self.type_im == 0:
            self.IM = self.IM1
        elif self.type_im == 2:
            self.IM = self.IM1 + 1j * self.IM2
        else:
            self.IM = self.IM1 * np.exp(2j * np.pi * np.random.uniform(size=self.IM1.shape))

# generate masks
        if self.mask_type == 0:  # iid mask
            self.phase_arg = np.random.uniform(size=(self.Na, self.Nb))
            self.mask = np.exp(2j * np.pi * self.phase_arg)

# generate 'b' = fft(im)
# num_mask2 = two different random masks
# num_mask3 = one id mask and one random mask
        self.Y = Nos_fft_num_mask3(self.IM, self.os_rate, self.mask)
        self.b = np.abs(self.Y)
        self.Z = self.b ** 2

        self.norm_Y = np.linalg.norm(self.b, 'fro')

        # initial gauss on ptycho_fft(IM)
        self.lambda_0 = np.random.uniform(size=self.Z.shape) * np.exp(2j*np.pi*np.random.uniform(size=self.Z.shape))


        # true mask
        self.mask = np.exp(2j * np.pi * self.phase_arg)

        # epoch counter
        self.update = 1

        # residual recorder
        self.resi_y_RAAR = np.zeros((self.MaxIter,), dtype='float')
        self.resi_y_APR = np.zeros((self.MaxIter,), dtype='float')
        self.resi_y_AAR = np.zeros((self.MaxIter,), dtype='float')
        self.resi_y_DR = np.zeros((self.MaxIter,), dtype='float')

        # relative error on image
        self.relative_y_RAAR = np.zeros((self.MaxIter,), dtype='float')
        self.relative_y_APR = np.zeros((self.MaxIter,), dtype='float')
        self.relative_y_AAR = np.zeros((self.MaxIter,), dtype='float')
        self.relative_y_DR = np.zeros((self.MaxIter,), dtype='float')

        # find (AA^*) = ..
        self.IM_ = np.ones((self.Na, self.Nb), dtype='complex')
        self.fft_IM_ = Nos_fft_num_mask3(self.IM_, self.os_rate, self.mask)
        self.nor_fac = Nos_ifft_num_mask3(self.fft_IM_, self.os_rate, self.mask)

        # beta is used in RAAR \in [0.5, 1)
        self.beta = 0.9

        self.lambda_t_RAAR = self.lambda_0
        self.lambda_t_APR = self.lambda_0
        self.lambda_t_AAR = self.lambda_0
        self.lambda_t_DR = self.lambda_0


    def one_step(self):
        # RAAR
        self.Py_x = P_Y(self.lambda_t_RAAR, self.b)
        self.Ry_x = 2 * self.Py_x - self.lambda_t_RAAR
        self.RxRy_x = 2 * P_X3(self.Ry_x, self.os_rate, self.mask, self.nor_fac) - self.Ry_x

        self.lambda_t_RAAR = self.beta / 2 * self.lambda_t_RAAR + self.beta / 2 * self.RxRy_x + \
                        (1 - self.beta) * self.Py_x
        self.x_new = Nos_ifft_num_mask3(self.lambda_t_RAAR, self.os_rate, self.mask) / self.nor_fac

        self.ee_RAAR = np.abs(self.IM.reshape(-1).conj().dot(self.x_new.reshape(-1))) / self.IM.reshape(-1).conj().dot(
            self.x_new.reshape(-1))
        self.rel_x_RAAR = np.linalg.norm(self.x_new * self.ee_RAAR - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')
        self.relative_y_RAAR[self.update - 1] = self.rel_x_RAAR
        self.res_x = np.linalg.norm(np.abs(self.b) - np.abs(self.lambda_t_RAAR), 'fro') / self.norm_Y
        self.resi_y_RAAR[self.update - 1] = self.res_x


        # APR
        self.Py_x= P_Y(self.lambda_t_APR, self.b)
        self.RxPy_x = 2 * P_X3(self.Py_x, self.os_rate, self.mask, self.nor_fac) - self.Py_x

        self.lambda_t_APR = 0.5 * self.lambda_t_APR + 0.5 * self.RxPy_x
        self.x_new = Nos_ifft_num_mask3(self.lambda_t_APR, self.os_rate, self.mask) / self.nor_fac

        self.ee_APR = np.abs(self.IM.reshape(-1).conj().dot(self.x_new.reshape(-1))) / self.IM.reshape(-1).conj().dot(
            self.x_new.reshape(-1))

        self.rel_x_APR = np.linalg.norm(self.x_new * self.ee_APR - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')
        self.relative_y_APR[self.update - 1] = self.rel_x_APR
        self.res_x = np.linalg.norm(np.abs(self.b) - np.abs(self.lambda_t_APR), 'fro') / self.norm_Y
        self.resi_y_APR[self.update - 1] = self.res_x



        # AAR
        self.Py_x = P_Y(self.lambda_t_AAR, self.b)
        self.Ry_x = 2 * self.Py_x - self.lambda_t_AAR
        self.RxRy_x = 2 * P_X3(self.Ry_x, self.os_rate, self.mask, self.nor_fac) - self.Ry_x
        self.lambda_t_AAR = 0.5 * self.lambda_t_AAR + 0.5 * self.RxRy_x
        self.x_new = Nos_ifft_num_mask3(self.lambda_t_AAR, self.os_rate, self.mask) / self.nor_fac

        self.ee_AAR = np.abs(self.IM.reshape(-1).conj().dot(self.x_new.reshape(-1))) / self.IM.reshape(-1).conj().dot(
            self.x_new.reshape(-1))
        self.rel_x_AAR = np.linalg.norm(self.x_new * self.ee_AAR - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')
        self.relative_y_AAR[self.update - 1] = self.rel_x_AAR

        self.res_x = np.linalg.norm(np.abs(self.b) - np.abs(self.lambda_t_AAR), 'fro') / self.norm_Y
        self.resi_y_AAR[self.update - 1] = self.res_x


        # DR
        self.Px_x_DR = P_X3(self.lambda_t_DR, self.os_rate, self.mask, self.nor_fac)
        self.Py_x_DR = P_Y(self.lambda_t_DR, self.b)
        self.RxPy_x_DR = 2 * P_X3(self.Py_x_DR, self.os_rate, self.mask, self.nor_fac) - self.Py_x_DR
        self.lambda_t_DR = 1 / (self.rho + 1) * self.lambda_t_DR + \
                        (self.rho - 1) / (self.rho + 1) * self.Px_x_DR + \
                        1 / (self.rho + 1) * self.RxPy_x_DR
        self.x_new_DR = Nos_ifft_num_mask3(self.lambda_t_DR, self.os_rate, self.mask) / self.nor_fac

        self.ee_DR = np.abs(self.IM.reshape(-1).conj().dot(self.x_new_DR.reshape(-1))) / self.IM.reshape(-1).conj().dot(
            self.x_new_DR.reshape(-1))
        self.rel_x_DR = np.linalg.norm(self.x_new_DR * self.ee_DR - self.IM, 'fro') / np.linalg.norm(self.IM, 'fro')
        self.relative_y_DR[self.update - 1] = self.rel_x_DR

        self.res_x = np.linalg.norm(np.abs(self.b) - np.abs(self.lambda_t_DR), 'fro') / self.norm_Y
        self.resi_y_DR[self.update - 1] = self.res_x



        print('update=%d rel_RAAR=%s rel_DR=%s \n rel_AAR=%s rel_APR=%s \n' % (
            self.update, '{:.4e}'.format(self.rel_x_RAAR),
            '{:.4e}'.format(self.rel_x_DR),
            '{:.4e}'.format(self.rel_x_AAR),
            '{:.4e}'.format(self.rel_x_APR)))

        self.update = self.update + 1

        return self.update, self.rel_x_RAAR, self.rel_x_APR, self.rel_x_AAR, self.rel_x_DR








def main():
    input_parameters = { 'pois_gau': 1,
                         'image_type': 'CiB_image',
                         'image_path': '/Users/zheqingzhang/Desktop/image_lib/Cameraman.png',
                         'image_path_real': '/Users/zheqingzhang/Desktop/image_lib/Cameraman.png',
                         'image_path_imag': '/Users/zheqingzhang/Desktop/image_lib/Barbara256.png',
                         'mask_type': 'IID',
                         'mask_delta': 0.5,
                         'MaxIter': 250,
                         'Toler':1e-7,
                         'rho': 0.5,
                         'os_rate': 2,
    }
    compare = RAAR_APR_AAR_phase_retrieval(input_parameters)
    Tol = input_parameters['Toler']
    Max_It = input_parameters['MaxIter']
    res_x = 1
    update = 1
    Iter=[]
    relative_y_RAAR = []
    relative_y_APR = []
    relative_y_AAR = []
    relative_y_DR = []

    while update < int(Max_It):
        update, rel_x_RAAR, rel_x_APR, rel_x_AAR, rel_x_DR = compare.one_step()
        Iter.append(update - 1)  # the output of count_DR is count_DR += 1
        relative_y_RAAR.append(rel_x_RAAR)
        relative_y_APR.append(rel_x_APR)
        relative_y_AAR.append(rel_x_AAR)
        relative_y_DR.append(rel_x_DR)


    plt.figure(0)
    plt.title('Error Plot')
    plt.xlabel('iter')
    plt.ylabel('error')
    plt.grid(True)

    plt.semilogy(Iter[0], relative_y_RAAR[0], 'ro-')
    plt.semilogy(Iter[0], relative_y_APR[0], 'b^:')
    plt.semilogy(Iter[0], relative_y_AAR[0], 'k*--')
    plt.semilogy(Iter[0], relative_y_DR[0], 'g*-')
    plt.legend(('RAAR 0.9', 'APR', 'AAR', 'DR 0.5'),
               loc='lower left')

    plt.semilogy(Iter[0:-1:5], relative_y_RAAR[0:-1:5], 'ro')
    plt.semilogy(Iter[0:-1:5], relative_y_APR[0:-1:5], 'b^')
    plt.semilogy(Iter[0:-1:5], relative_y_AAR[0:-1:5], 'k*')
    plt.semilogy(Iter[0:-1:5], relative_y_DR[0:-1:5], 'g*')

    plt.semilogy(Iter, relative_y_RAAR, 'r-')
    plt.semilogy(Iter, relative_y_APR, 'b:')
    plt.semilogy(Iter, relative_y_AAR, 'k--')
    plt.semilogy(Iter, relative_y_DR, 'g-')

    save_path_1 = '/Users/zheqingzhang/Documents/phase retrieval/Blind_Ptychography_GUI/workspace'
    save_path = os.path.join(save_path_1, 'error plot6.png')
    plt.savefig(save_path)
    plt.close(0)



main()