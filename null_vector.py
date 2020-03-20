import numpy as np



class null_vector:
    def __init__(self,m, n, x, numI):

        self.n = n
        self.m = m
        self.im = x
        self.sen_matrix = np.random.normal(0, 1, (self.m, self.n)) + np.random.normal(0, 1, (self.m, self.n))*1j
        self.b = np.abs(self.sen_matrix.dot(self.im))
        self.I = numI


        self.bpower2 = (self.b) ** 2
        self.bpower2.sort()
        self.threshold_chen = self.bpower2[self.I-1]
        self.threshold_chen = np.sqrt(self.threshold_chen)
        self.len_vec_z = len(self.bpower2)
        self.trQX = np.zeros(self.len_vec_z)
        for i in range(self.len_vec_z):
            self.trQX[i] = (np.sum([self.bpower2[-(i + 2):-1]]) / ((i + 1)**0.52))

        self.optimal_b = max(self.trQX)
        self.max_location = [self.maxlocation for self.maxlocation, self.j in enumerate(self.trQX) if
                             self.j == self.optimal_b]

        self.threshold_b = np.sqrt(self.bpower2[self.m - self.max_location[0]])
        print(self.max_location)
        self.ind = np.zeros([self.m,self.m], dtype=bool)
        self.ind_chen = np.zeros([self.m, self.m], dtype=bool)
        for i in range(self.m):
            if self.b[i] > self.threshold_chen:   # modify
                self.ind_chen[i, i] = True
            if self.b[i] > self.threshold_b:   # modify
                self.ind[i, i] = True



        self.q, self.r = np.linalg.qr(self.sen_matrix)


        self.qrim = self.r.dot(self.im)
        self.magnitude = np.linalg.norm(self.qrim, 2)

        self.itermatrix = self.ind.dot(self.q)
        self.itermatrix_chen = self.ind_chen.dot(self.q)

        self.iter = 0

        self.ini = np.ones(np.size(self.im),dtype=float)
        self.y = self.ini
        self.y_chen = self.ini
        while self.iter < 250:
            self.interm = self.itermatrix.dot(self.y)
            self.y_plus1 = self.itermatrix.conj().T.dot(self.interm)
            self.y_plus1 = self.y_plus1*self.magnitude/np.linalg.norm(self.y_plus1, 2)

            self.ee_im = np.abs(self.qrim.conj().dot(self.y_plus1)) / self.qrim.conj().dot(
                self.y_plus1)
            self.rel_im = np.linalg.norm(self.ee_im * self.y_plus1 - self.qrim, 2) / np.linalg.norm(self.qrim, 2)
            self.y = self.y_plus1
            #
            self.interm_chen = self.itermatrix_chen.dot(self.y)
            self.y_plus1_chen = self.itermatrix_chen.conj().T.dot(self.interm_chen)
            self.y_plus1_chen = self.y_plus1_chen * self.magnitude / np.linalg.norm(self.y_plus1_chen, 2)

            self.ee_im_chen = np.abs(self.qrim.conj().dot(self.y_plus1_chen)) / self.qrim.conj().dot(
                self.y_plus1_chen)
            self.rel_im_chen = np.linalg.norm(self.ee_im_chen * self.y_plus1_chen - self.qrim, 2) / np.linalg.norm(self.qrim, 2)
            self.y_chen = self.y_plus1_chen
            self.iter +=1



            print('iter={} rel_im ={:.4e} rel_im_chen={:.4e}'.format(self.iter, self.rel_im**2, self.rel_im_chen**2))

        self.x_null = np.linalg.inv(self.r).dot(self.y_plus1)
        self.x_null = self.x_null * np.linalg.norm(self.im, 2)/np.linalg.norm(self.x_null, 2)
        self.ee_im = np.abs(self.im.conj().dot(self.x_null)) / self.im.conj().dot(
            self.x_null)
        self.rel_im = np.linalg.norm(self.ee_im * self.x_null - self.im, 2) / np.linalg.norm(self.im, 2)
        print('Endrel_im ={:.4e}'.format(self.rel_im ** 2))

        print(numI, self.max_location)

n = 15
L = 40
m = L * n
alpha = 0.6
numI = round((n**(1-alpha))*(m**alpha))


x = np.random.normal(0,1,n) + 1j*np.random.normal(0,1,n)

null_vector(m, n, x, numI)