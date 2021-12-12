# _*_ coding:UTF-8 _*_
# developer: yaoji
# Time: 11/12/20204:18 PM

import numpy as np
from builtins import object
from scipy import signal
from scipy import spatial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import pyMRAW
import pyidi
from scipy import linalg
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io
import imageio
import pandas as pd
import pylab as pl

class Model(object):

    def __init__(
        self,
        sampling_frequency,
        Order,
        Data,
        Method,
        data_path='Not needed',
        sf='auto',
        filetype='matLMS'
    ):

        self.sampling_frequency = sampling_frequency
        self.order = Order
        self.Data = Data
        self.Method = Method
        self.cache = []

        self.displacements = Data
        self.filetype = filetype
        self.data_path = data_path
        self.sf = sf

        if self.data_path != 'Not needed':
            if self.filetype == 'cih':
                """
                Loads image data (.cih file) by using the pyMRAW Toolbox.
                pyMRAW: https://github.com/ladisk/pyMRAW
                """
                self.video = pyidi.pyIDI(self.data_path)                         # video_path = '/Users/milesjudd/PycharmProjects/OMA/mraw/low_rate_sof.cih'
                self.mraw, self.info = pyMRAW.load_video(self.data_path)
                if self.sf == 'auto':
                    self.sf = float(self.info['Record Rate(fps)'])

            elif self.filetype == 'mat':
                """
                Loads time history data (.mat file) from MATLAB.
                """
                mat = scipy.io.loadmat(self.data_path)      # /Users/milesjudd/PycharmProjects/OMA/mat/OMA_Run3.mat
                self.displacements = mat['Signal']


            elif self.filetype == 'matLMS':
                """
                Loads time history data (.mat file) obtained from the LMS Test.Lab. When time history data saved as .mat file from LMS, no further formatting of the .mat file is required!
                """
                mat = scipy.io.loadmat(self.data_path)      # /Users/milesjudd/PycharmProjects/OMA/mat/OMA_Run3.mat
                data = mat['Signal']

                y_data = data['y_values']
                y_dataa = y_data[0]
                y_dataaa = y_dataa[0]
                y_dataaaa = y_dataaa[0]
                y_dataaaaa = y_dataaaa[0]
                y_dataaaaaa = y_dataaaaa[0]
                self.datashape = y_dataaaaaa.shape

                x_data = data['x_values']
                x_dataa = x_data[0]
                x_dataaa = x_dataa[0]
                x_dataaaa = x_dataaa[0]
                x_dataaaaa = x_dataaaa[0]
                x_dataaaaaa = x_dataaaaa[0]
                x_dataaaaaaa = x_dataaaaaa[0]
                start_value = x_dataaaaaaa[0]
                # print(1 / start_value)
                sf = y_dataaaaaa.shape[0] // 30  # 12834
                t = 30
                self.displacements = np.transpose(y_dataaaaaa)
                self.start_value = x_dataaaaaaa[0]
                if self.sf == 'auto':
                    increment0 = x_dataaaaa[1]
                    increment1 = increment0[0]
                    increment = increment1[0]
                    self.sf = 1 / increment

            else:
                print("Data-Filetype not defined! Choose filetype: 'cih' / 'mat' / 'matLMS'")

        if Method == 'cov_SSI':
            self.cache = self.cov_SSI()
        elif Method == 'ITD':
            self.cache = self.ITD()
        elif Method == 'DD_SSI':
            self.cache = self.DD_SSI()


# SSI-Method

    def SVD_raw_Daten(self, ax):
        Sampling_frequency = self.sampling_frequency
        Data = self.Data
        channel = Data.shape[0]

        psd = np.zeros((Sampling_frequency // 2 + 1) * Data.shape[0] * Data.shape[0], dtype=complex).reshape(
            Sampling_frequency // 2 + 1, Data.shape[0], Data.shape[0])
        S = np.zeros(Sampling_frequency // 2 + 1)
        for i in range(channel):
            for j in range(channel):
                f, Pxy = signal.csd(Data[i], Data[j], Sampling_frequency, nperseg=Sampling_frequency)
                psd[:, i, j] = np.abs(Pxy)

        for i in range(psd.shape[0]):
            u, s, v = np.linalg.svd(psd[i, :, :])
            S[i] = 20 * np.log10(np.abs(s[0]))
            
        S -= np.min(S)
        S /= np.max(S) / np.max(self.order) * 2
        ax.plot(f, S, label='Mode indicator Function')
        legend = plt.legend(loc='lower right', fontsize=15)
        plt.grid()
        plt.show()

    def R_dach(self, Data, order):
        """

        :param Daten:
        :param ordnung:
        :return:
        """
        c = Data.shape[0]
        N = Data.shape[1]
        set = 2 * order
        size = c * set * c

        R_dach = np.zeros(size).reshape(c, size // c)
        arange = np.arange(1, 2 * order + 1, 1)

        for i in arange:
            R_dach[:, (set - i) * c:(set - i + 1) * c] = 1 / (N - 1) * Data[:, :N - i].dot(Data[:, i:N].T)

        return R_dach, c


    def Toeplitz_matrix(self, R_dach, c, lag, order):
        """

        :param R_dach:
        :param c:
        :param lag:
        :param ordnung:
        :return:
        """

        size = c ** 2 * order ** 2  # Matrix size
        set = 2 * order  # R_dach hat x set von c*c matrix

        toeplitz_matrix = np.zeros(size).reshape(c * order, c * order)
        arange = np.arange(1, order + 1, 1)

        for i in arange:
            toeplitz_matrix[(i - 1) * c:i * c, :] = R_dach[:, (set - i + 1 - order - lag) * c:(set - i + 1 - lag) * c]
        return toeplitz_matrix

    def cov_SSI(self):
        order = self.order
        sampling_frequency = self.sampling_frequency
        Data = self.Data

        Nodes = Data.shape[0]
        Psi_m_matrix = np.zeros(Nodes ** 2 * order ** 2, dtype=complex).reshape(Nodes * order, Nodes * order)
        Frequency_natural = np.zeros(order ** 2 * Nodes).reshape(order, order * Nodes)
        Frequency_damped = np.zeros(order ** 2 * Nodes).reshape(order, order * Nodes)
        Damping_ratio = np.zeros(order ** 2 * Nodes).reshape(order, order * Nodes)

        arange = np.arange(2, order + 1, 1)
        for i in arange:
            R_dach, c = self.R_dach(Data, i)  # (check)
            toeplitz_matrix_1 = self.Toeplitz_matrix(R_dach, c, lag=0, order=i)  # Toeplitz matrix T1 (check)
            u, s, v = np.linalg.svd(toeplitz_matrix_1)
            toeplitz_matrix_2 = self.Toeplitz_matrix(R_dach, c, lag=1, order=i)  # Toeplitz matrix T2 (check)

            Matrix_O_plus = np.sqrt(np.diagflat(s ** -1)).dot(u.T)  # (check)
            Matrix_Tau_plus = (v.T).dot(np.sqrt(np.diagflat(s ** -1)))
            Matrix_C = u.dot(np.sqrt(np.diagflat(s)))[0:Nodes, :]  # Matrix C (check)

            Matrix_A = np.linalg.multi_dot([Matrix_O_plus, toeplitz_matrix_2, Matrix_Tau_plus])  # Matrix A
            eigenvalue_A, eigenvector_A = np.linalg.eig(Matrix_A)

            arange = np.arange(1, i * Nodes + 1, 1)
            for j in arange:
                Psi_m = Matrix_C.dot(eigenvector_A[:, (j - 1)])
                Psi_m_matrix[(i - 1) * Nodes:i * Nodes, (j - 1)] = Psi_m

            Lambda = np.zeros(i * Nodes, dtype=complex)
            for j in arange:
                Lambda[j - 1] = np.log([eigenvalue_A[(j - 1)]]) * sampling_frequency

            frequeny_natural = np.zeros(i * Nodes)
            for j in arange:
                frequeny_natural[j - 1] = np.abs([Lambda[j - 1] / 2 / np.pi])

            frequeny_damped = np.zeros(i * Nodes)
            for j in arange:
                frequeny_damped[j - 1] = np.imag([Lambda[j - 1] / 2 / np.pi])

            damping_ratio = np.zeros(i * Nodes)
            for j in arange:
                damping_ratio[j - 1] = -np.real(Lambda[j - 1] / np.abs([Lambda[j - 1]]))

            Frequency_natural[i - 1, 0:i * Nodes] = frequeny_natural  # cache
            Frequency_damped[i - 1, 0:i * Nodes] = frequeny_damped  # cache
            Damping_ratio[i - 1, 0:i * Nodes] = damping_ratio  # cache

        Psi_m_Matrix = Psi_m_matrix[:, :]
        self.cache = {'Frequency_natural': Frequency_natural, 'Frequency_damped': Frequency_damped,
                      'Damping_ratio': Damping_ratio, 'Psi_m_Matrix': Psi_m_Matrix,
                      'sampling_frequency': sampling_frequency,
                      'order': order, 'Data': Data}

        return self.cache


    def Stablisation_Diagramm(self, plot_svd, verbose):
        """

        :param cache:
        :return:
        """
        order = self.order
        Data = self.Data
        Nodes = Data.shape[0]
        cache = self.cache
        Frequency_natural = cache['Frequency_natural']
        Frequency_damped = cache['Frequency_damped']
        Damping_ratio = cache['Damping_ratio']
        Psi_m_Matrix = cache['Psi_m_Matrix']
        Method = self.Method
        SD_option = verbose

        if Method == 'ITD':
            Frequency_stability = np.zeros(order * Frequency_natural.shape[1]).reshape(order,
                                                                                       Frequency_natural.shape[1])
            Damping_ratio_stability = np.zeros(order * Frequency_natural.shape[1]).reshape(order,
                                                                                           Frequency_natural.shape[1])
            MAC_stability_all = np.zeros(order * Frequency_natural.shape[1]).reshape(order, Frequency_natural.shape[1])
        else:
            Frequency_stability = np.zeros(order * order * Nodes).reshape(order, order * Nodes)
            Damping_ratio_stability = np.zeros(order * order * Nodes).reshape(order, order * Nodes)
            MAC_stability_all = np.zeros(order * order * Nodes).reshape(order, order * Nodes)

        # Frequency compare     (check)
        arange_order_counter = np.arange(2, order, 1)
        for order_counter in arange_order_counter:
            if Method == 'ITD':
                arange_element_counter = np.arange(0, Frequency_natural.shape[1], 1)
            else:
                arange_element_counter = np.arange(1, order_counter * Nodes, 1)

            for element_counter in arange_element_counter:
                A = np.abs((Frequency_natural[order_counter - 1, :order_counter * Nodes] -
                            Frequency_natural[order_counter, element_counter]))
                A = np.where(A > 2, 2000, A)  # Begrenzen die Toleranz bis zu 2 Hz
                Frequency_stable = np.divide(A, Frequency_natural[order_counter - 1, :order_counter * Nodes])  # (check)

                if any(y < 0.01 for y in Frequency_stable):
                    Frequency_stability[order_counter, element_counter] = 1
                    index_freq = np.where(Frequency_stable < 0.01)
                    # damping
                    for damping_index in index_freq:
                        B = np.abs(Damping_ratio[order_counter - 1, damping_index] - Damping_ratio[
                            order_counter, element_counter])
                        Damping_stable = np.divide(B, Damping_ratio[order_counter - 1, damping_index])

                        if np.any(Damping_stable < 0.05) and np.any(
                                Damping_ratio[order_counter, element_counter] > 0):
                            Damping_ratio_stability[order_counter, element_counter] = 1  # (check)

                            Psi_low = Psi_m_Matrix[
                                      (order_counter - 1) * Data.shape[0]:order_counter * Data.shape[0],
                                      damping_index]
                            Psi_high = Psi_m_Matrix[
                                       order_counter * Data.shape[0]:(order_counter + 1) * Data.shape[0],
                                       element_counter]
                            numerator = np.abs((np.conj(Psi_low).T).dot(Psi_high)) ** 2
                            denominator = (np.conj(Psi_low).T).dot(Psi_low) * (np.conj(Psi_high).T).dot(Psi_high)
                            mac_stable = np.divide(numerator, denominator)
                            if np.any(1 - mac_stable < 0.02):
                                MAC_stability_all[order_counter, element_counter] = 1
        if Method == 'ITD':
            stable_matrix_all = np.zeros(order * Frequency_natural.shape[1]).reshape(order, Frequency_natural.shape[1])
            stable_matrix_freq = np.zeros(order * Frequency_natural.shape[1]).reshape(order, Frequency_natural.shape[1])
        else:
            stable_matrix_all = np.zeros(order * order * Nodes).reshape(order, order * Nodes)
            stable_matrix_freq = np.zeros(order * order * Nodes).reshape(order, order * Nodes)

        stable_matrix_all += Frequency_natural * MAC_stability_all
        stable_matrix_freq += Frequency_natural * Frequency_stability

        if verbose == True:
            points_x = stable_matrix_all[stable_matrix_all > 0]
            points_y = np.nonzero(stable_matrix_all > 0)[0]
            stable_points = np.column_stack((points_x, points_y))
            points_x_verbose =stable_matrix_freq[stable_matrix_freq > 0]
            points_y_verbose = np.nonzero(stable_matrix_freq > 0)[0]
            fig, ax = plt.subplots()
            ax.plot(points_x_verbose, points_y_verbose, '.', label='Frequency Stable Points', marker='+', color="#b3b300")
            ax.plot(points_x, points_y, '.', label='MAC Stable Points', color="#FF8000")
            plt.suptitle('Stabilisation Diagramm', fontsize=20)
            plt.xlabel('Frequency (Hz)', fontsize=20)
            plt.ylabel('Order', fontsize=20)

        elif verbose == False:
            points_x = stable_matrix_all[stable_matrix_all > 0]
            points_y = np.nonzero(stable_matrix_all > 0)[0]
            stable_points = np.column_stack((points_x, points_y))
            fig, ax = plt.subplots()
            ax.plot(points_x, points_y, '.', label='Stable Points', color="#FF8000")
            plt.suptitle('Stabilisation Diagramm', fontsize=20)
            plt.xlabel('Frequency (Hz)', fontsize=20)
            plt.ylabel('Order', fontsize=20)

        if plot_svd == True:
            self.SVD_raw_Daten(ax=ax)

        return fig, ax, stable_points, self.cache


    def plot_Mode(self, freq_dr_esf, Geometry):
        """

        :param Freq_Dr_ESF:
        :param messung_position:
        :param number_of_points:
        :return:
        """
        Data = self.Data
        number_of_points = freq_dr_esf.size // (Data.shape[0] + 2)
        Freq_Dr_ESF = freq_dr_esf.reshape(number_of_points, freq_dr_esf.size // number_of_points)

        fig, axs = plt.subplots(number_of_points, 1, figsize=(8, 12), constrained_layout=True)
        for i in range(number_of_points):
            Freq_Dr_ESF[i, 2::] = Freq_Dr_ESF[i, 2::] / np.max(Freq_Dr_ESF[i, 2::])  # Set the max move to 1
            axs[i].plot(Geometry, np.array(Freq_Dr_ESF[i, 2::]))
            f2 = interp1d(Geometry, np.array(Freq_Dr_ESF[i, 2::]), kind='quadratic')
            xnew = np.linspace(Geometry[0], Geometry[4], num=201, endpoint=True)
            axs[i].plot(xnew, f2(xnew), '--')
            axs[i].set_title(
                'mode %i' % (i + 1) + '   Natural Frequency: %1.11f' % Freq_Dr_ESF[i, 0] + "  Damping Ratio: %1.11f" %
                Freq_Dr_ESF[i, 1])
            axs[i].set_xlabel('distance (cm)')
        plt.show()

    def DD_SSI(self):
        order = self.order
        sampling_frequency = self.sampling_frequency
        Data = self.Data

        Nodes = Data.shape[0]
        Psi_m_matrix = np.zeros(Nodes ** 2 * order ** 2, dtype=complex).reshape(Nodes * order, Nodes * order)
        Frequency_natural = np.zeros(order ** 2 * Nodes).reshape(order, order * Nodes)
        Frequency_damped = np.zeros(order ** 2 * Nodes).reshape(order, order * Nodes)
        Damping_ratio = np.zeros(order ** 2 * Nodes).reshape(order, order * Nodes)

        channel = Data.shape[0]

        arange = np.arange(1, order + 1, 1)
        for i in arange:
            length = Data.shape[1] - 2 * i
            size = channel * i * length * 2
            Block_Hankel_matrix = np.zeros(size).reshape(channel * i * 2, length)
            arange_sub = np.arange(1, i * 2 + 1, 1)

            for j in arange_sub:
                Block_Hankel_matrix[(j - 1) * channel:j * channel] = Data[:, j - 1:j - 1 + length]

            Block_Hankel_matrix *= 1 / np.sqrt(length)
            # ==================================================
            # compute Projection
            Y_p = Block_Hankel_matrix[0:channel * i, :]
            Y_f = Block_Hankel_matrix[channel * i::, :]
            Matrix_Y_i_i = Block_Hankel_matrix[channel * i:channel * (i + 1), :]

            Y_p_plus = Block_Hankel_matrix[0:channel * (i + 1), :]
            Y_f_minus = Block_Hankel_matrix[channel * (i + 1)::, :]

            Y_p_inverse = np.linalg.pinv(Y_p.dot(Y_p.T))
            Y_p_f = Y_f.dot(Y_p.T)
            Projection_i = np.linalg.multi_dot([Y_p_f, Y_p_inverse, Y_p])

            Y_p_i_minus_1_inverse = np.linalg.inv(Y_p_plus.dot(Y_p_plus.T))
            Y_p_f_i_minus_1 = Y_f_minus.dot(Y_p_plus.T)
            Projection_i_minus_1 = np.linalg.multi_dot([Y_p_f_i_minus_1, Y_p_i_minus_1_inverse, Y_p_plus])

            # ==================================================
            # Get Observation matrix O and Kalman filter state S
            u, s, vT = np.linalg.svd(Projection_i)
            Observation_Matrix_i = u.dot(np.sqrt(np.diagflat(s ** -1)))
            Kalman_filter_state_i = (np.linalg.inv(Observation_Matrix_i)).dot(Projection_i)

            Observation_Matrix_i_upper = Observation_Matrix_i[0:-1 * channel, :]
            Observation_Matrix_i_lower = Observation_Matrix_i[channel::, :]
            Kalman_filter_state_i_plus_1 = (np.linalg.pinv(Observation_Matrix_i_upper)).dot(Projection_i_minus_1)

            # ==================================================
            # Use the First Method
            Matrix_AC_1 = np.vstack([Kalman_filter_state_i_plus_1, Matrix_Y_i_i])
            Matrix_AC = Matrix_AC_1.dot(np.linalg.pinv(Kalman_filter_state_i))
            Matrix_A = Matrix_AC[0:-1 * channel, :]
            # Matrix_A = np.linalg.pinv(Observation_Matrix_i_upper).dot(Observation_Matrix_i_lower)
            Matrix_C = Matrix_AC[-1 * channel::, :]

            eigenvalue_A, eigenvector_A = np.linalg.eig(Matrix_A)
            # eigenvalue_A = eigenvalue_A ** (1. / (i+1))
            # ==================================================

            arange = np.arange(1, i * Nodes + 1, 1)
            for j in arange:
                Psi_m = Matrix_C.dot(eigenvector_A[:, (j - 1)])
                Psi_m_matrix[(i - 1) * Nodes:i * Nodes, (j - 1)] = Psi_m

            Lambda = np.zeros(i * Nodes, dtype=complex)
            for j in arange:
                Lambda[j - 1] = np.log([eigenvalue_A[(j - 1)]]) * sampling_frequency

            frequeny_natural = np.zeros(i * Nodes)
            for j in arange:
                frequeny_natural[j - 1] = np.abs(Lambda[j - 1]) / 2 / np.pi

            frequeny_damped = np.zeros(i * Nodes)
            for j in arange:
                frequeny_damped[j - 1] = np.imag([Lambda[j - 1] / 2 / np.pi])

            damping_ratio = np.zeros(i * Nodes)
            for j in arange:
                damping_ratio[j - 1] = -np.real(Lambda[j - 1] / np.abs([Lambda[j - 1]]))

            Frequency_natural[i - 1, 0:i * Nodes] = frequeny_natural  # cache
            Frequency_damped[i - 1, 0:i * Nodes] = frequeny_damped  # cache
            Damping_ratio[i - 1, 0:i * Nodes] = damping_ratio  # cache

        Psi_m_Matrix = Psi_m_matrix[:, :]
        self.cache = {'Frequency_natural': Frequency_natural, 'Frequency_damped': Frequency_damped,
                      'Damping_ratio': Damping_ratio, 'Psi_m_Matrix': Psi_m_Matrix,
                      'sampling_frequency': sampling_frequency,
                      'order': order, 'Data': Data}
        return self.cache

    '''
    ITD-Algorithm 
    '''

    def ITD(self):
        order = self.order
        sampling_frequency = self.sampling_frequency
        Data = self.Data

        Nodes = Data.shape[0]
        Psi_m_Matrix = np.zeros(Nodes ** 2 * order, dtype=complex).reshape(Nodes * order, Nodes)
        Frequency_natural = np.zeros(order * Nodes).reshape(order, Nodes)
        Frequency_damped = np.zeros(order * Nodes).reshape(order, Nodes)
        Damping_ratio = np.zeros(order * Nodes).reshape(order, Nodes)

        begin = 10
        length = Data.shape[1] - begin - order - 1
        Y1 = np.zeros(Data.shape[0] * 2 * length).reshape(Data.shape[0] * 2, length)
        Y1[0:Data.shape[0], :] = Data[:, begin:begin + length]
        Y1[Data.shape[0]:Data.shape[0] * 2, :] = Data[:, begin + 1:begin + length + 1]

        Y2 = np.zeros(Data.shape[0] * 2 * length).reshape(Data.shape[0] * 2, length)

        arange = np.arange(1, order + 1, 1)
        for i in arange:
            Y2[0:Data.shape[0], :] = Data[:, begin + i:begin + i + length]
            Y2[Data.shape[0]:Data.shape[0] * 2, :] = Data[:, begin + i + 1:begin + i + length + 1]

            A1_1 = Y2.dot(Y1.transpose())
            A1_2 = np.linalg.inv(Y1.dot(Y1.transpose()))
            A1 = np.dot(A1_1, A1_2)

            A2_1 = Y2.dot(Y2.transpose())
            A2_2 = np.linalg.inv(Y1.dot(Y2.transpose()))
            A2 = np.dot(A2_1, A2_2)

            Matrix_A = A1

            Lambda, mod = np.linalg.eig(Matrix_A)
            Lambda = Lambda ** (1. / i)

            bet1 = np.real(Lambda)
            gam1 = np.imag(Lambda)
            a1 = -sampling_frequency / 2 * np.log(bet1 ** 2 + gam1 ** 2)
            b1 = sampling_frequency * np.arctan(gam1 / bet1)
            omg_ib1 = np.sqrt(a1 ** 2 + b1 ** 2)
            cita = a1 / omg_ib1

            Frequency_natural[i - 1, :] = omg_ib1[0::2] / 2 / np.pi
            Psi_m_Matrix[(i - 1) * Nodes:i * Nodes, :] = np.array(mod)[0:Nodes, 0::2]
            Damping_ratio[i - 1, :] = cita[0::2]

        cache = {'Frequency_natural': Frequency_natural, 'Frequency_damped': Frequency_damped,
                 'Damping_ratio': Damping_ratio, 'sampling_frequency': sampling_frequency,
                 'Psi_m_Matrix': Psi_m_Matrix, 'order': order, 'Data': Data}
        return cache



    '''
    FDD-Algorithm (Basic principle: Singular value decomposition (SVD) of the spectral density (SD) matrix)

    This code will perform a Frequency Domain Decomposition (FDD) of a given signal.

    The basic steps are the following:
    1. Import the signal
    2. Compute the Power Spectral Density
    3. Perform a Singular Value Decomposition of the Power Spectral Density Matrix
    4. Pick Peaks
    '''

    def FDD(self, npsg = 'auto', plot = True):
        """
        Performs a Frequency Domain Decomposition of the data. Resulting Single Values are plotted and peaks labeled.

        :param npsg:
        :param plot:
        :return:
        """

        if npsg == 'auto':
            npsg = 2
            while npsg*2 < self.sf:
                npsg = npsg * 2         # Window size

        # Compute CPSD in z direction
        f = np.zeros((self.displacements.shape[0], self.displacements.shape[0], npsg // 2 + 1), dtype=float)
        Pxy = np.zeros((self.displacements.shape[0], self.displacements.shape[0], npsg // 2 + 1), dtype=float)
        for i, points_i in enumerate(self.displacements):
            for j, points_j in enumerate(self.displacements):
                f[i, j, :], Pxy[i, j, :] = signal.csd(points_i[:], points_j[:], self.sf, nperseg=npsg)

        # Compute Single Value Decomposition of Power Spectral Density Matrix
        s1 = np.zeros((npsg // 2 + 1), dtype=float)
        s2 = np.zeros((npsg // 2 + 1), dtype=float)
        s3 = np.zeros((npsg // 2 + 1), dtype=float)
        s4 = np.zeros((npsg // 2 + 1), dtype=float)
        self.ms = np.zeros((npsg // 2 + 1, self.displacements.shape[0], self.displacements.shape[0]), dtype=float)

        for x in range(npsg // 2 + 1):
            u, s, vh = linalg.svd(Pxy[:, :, x], full_matrices=True)  # performs a SVD of a 2D-Matrix "PSD"
            s1[x] = s[0]  # 1st eigenvalues
            s2[x] = s[1]  # 2nd eigenvalues
            s3[x] = s[2]  # 3th eigenvalues
            s4[x] = s[3]  # 4th eigenvalues
            self.ms[x, :, :] = u[:, :]

        if plot == True:
            # Plot normalised Single Values over frequency
            plt.figure(figsize=(15, 5))
            for i in range(15):
                for j in range(15):
                    plt.semilogy(f[i, j, :], s1 / max(s1))
                    plt.semilogy(f[i, j, :], s2 / max(s1))
                    plt.semilogy(f[i, j, :], s3 / max(s1))
                    plt.semilogy(f[i, j, :], s4 / max(s1))

            # Find and label peaks in SV-Plot
            plt.xlabel('frequency [Hz]')
            self.peaks, _ = signal.find_peaks(s1, wlen=500, prominence=0.1,
                                         distance=30)  # threshold=0.1, distance=5, prominence=0.2, width = 5
            plt.semilogy(f[0, 0, self.peaks], s1[self.peaks] / max(s1), "x")
            plt.ylabel('Single Value')
            for x, y in zip(f[0, 0, self.peaks], s1[self.peaks] / max(s1)):
                pl.text(x, y, str(int(x)), color="red", fontsize=8)
            pl.margins(0.1)
            plt.show()

        else:
            self.peaks, _ = signal.find_peaks(s1, wlen=500, prominence=0.1,
                                         distance=30)  # threshold=0.1, distance=5, prominence=0.2, width = 5



    def plotms(self, peaknr, channels, interpolate = True):
        """
        Plots the Modeshape for a specific frequency

        :param peaknr:
        :param channels:
        :param interpolate:
        :return:
        """
        self.FDD(plot = False)
        y = np.zeros(len(channels))
        for i in range(len(channels)):
            y[i] = self.ms[self.peaks[peaknr], channels[i], 0]
        plt.title('Mode Shape at ' + str(self.peaks[peaknr]) + 'Hz')
        plt.xlabel('Channel')

        if interpolate == True:
            xnew = np.linspace(0, len(channels)-1, num=len(channels)*4, endpoint=True)
            x = np.arange(len(channels))
            print(x)
            f = interp1d(x, y, kind='cubic')
            plt.plot(y, label='Line 1')
            plt.plot(x, y, 'o', xnew, f(xnew), '--')
            plt.legend(['linear','data', 'cubic'], loc='best')
            plt.show()
        else:
            plt.plot(y)
            plt.show()

    def mac(self, peaknr = 'All'):
        """
        Plots the Modal Assurance Criterion (MAC) for the identified peaks.

        :param peaknr:
        :return:
        """

        self.FDD(plot = False)

        if peaknr == 'All':
            MAC = np.zeros((self.peaks.size, self.peaks.size), dtype=float)
            for i, modes_i in enumerate(self.peaks):
                for j, modes_j in enumerate(self.peaks):
                    print(j)
                    print(modes_j)
                    print(self.peaks)
                    MAC[i,j] = abs(np.transpose(np.float128(self.ms[modes_i, :, 0])).dot(np.float128(self.ms[modes_j, :, 0])))**2 / ((np.transpose(np.float128(self.ms[modes_i, :, 0])).dot(np.float128(self.ms[modes_i, :, 0]))) * (np.transpose(np.float128(self.ms[modes_j, :, 0])).dot(np.float128(self.ms[modes_j, :, 0]))))
            plt.title('Modal Assurance Criterion for all Peak-Frequencies: ' + str(self.peaks) + ' [Hz]')
            plt.imshow(MAC)
            plt.show()

        else:
            MAC = np.zeros((len(peaknr), len(peaknr)), dtype=float)
            newpeaks = np.zeros(len(peaknr), dtype=int)
            for i in range(len(peaknr)):
                newpeaks[i] = self.peaks[peaknr[i]]
            for i, modes_i in enumerate(newpeaks):
                for j, modes_j in enumerate(newpeaks):
                    print(j)
                    print(modes_j)
                    print(newpeaks)
                    MAC[i,j] = abs(np.transpose(np.float128(self.ms[modes_i, :, 0])).dot(np.float128(self.ms[modes_j, :, 0])))**2 / ((np.transpose(np.float128(self.ms[modes_i, :, 0])).dot(np.float128(self.ms[modes_i, :, 0]))) * (np.transpose(np.float128(self.ms[modes_j, :, 0])).dot(np.float128(self.ms[modes_j, :, 0]))))
            plt.title('Modal Assurance Criterion for frequencies: ' + str(newpeaks) + ' [Hz]')
            plt.imshow(MAC)
            plt.show()

        return MAC

        
class choose_points:
    """

    """
    def __init__(self, points, fig, cache, ax):
        self.fig = fig
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.current_x = 0
        self.current_y = 0
        self.Freq_Dr_ESF = np.empty((1, 7))
        self.stable_points = points
        self.cache = cache
        self.count = 0
        self.ax = ax
        self.index_lib = np.empty((1, 2))

        self.ordnung_and_freq = []


    def __call__(self, event):
        Frequency_natural = self.cache['Frequency_natural']
        Damping_ratio = self.cache['Damping_ratio']
        Psi_m_Matrix = self.cache['Psi_m_Matrix']
        Daten = self.cache['Data']


        if event.button == 1:
            # find the point in Stable Diagram
            current_index = np.concatenate((event.xdata, event.ydata), axis=None)
            order_and_freq = self.stable_points[spatial.KDTree(self.stable_points).query(current_index)[1]]           # [frequency, ordnung]

            temp_order = int(round(order_and_freq[1]))
            temp_index_frequency = int((np.where(Frequency_natural[temp_order, :] == order_and_freq[0]))[0])

            picked_dr = Damping_ratio[temp_order, temp_index_frequency]
            picked_esf = np.real(
                Psi_m_Matrix[temp_order * Daten.shape[0]:temp_order * Daten.shape[0] + 5, temp_index_frequency])

            # check whether this point was already chosen, if not, take it
            if (not np.isin(order_and_freq[0], self.index_lib)):
                current = np.concatenate((np.array(order_and_freq[0]), picked_dr, picked_esf), axis=None)
                current = current.reshape((1, len(current)))
                self.Freq_Dr_ESF = np.vstack((self.Freq_Dr_ESF, current))
                temp_order_frequency = np.array([[temp_order, order_and_freq[0]]])
                self.index_lib = np.append(self.index_lib, temp_order_frequency, axis=0)
                self.count += 1

                # plot the point
                self.ax.scatter(order_and_freq[0], temp_order, color="#FF0066")
                self.fig.canvas.draw()

        if event.button == 3:
            if self.count != 0:
                # find the point in Stable Diagram
                current_index = np.concatenate((event.xdata, event.ydata), axis=None)
                order_and_freq = self.stable_points[spatial.KDTree(self.stable_points).query(current_index)[1]]
                temp_order = int(round(order_and_freq[1]))

                # if this point is already exsist, delete it
                if (np.isin(order_and_freq[0], self.Freq_Dr_ESF[:,0])):
                    row_index = np.where(self.Freq_Dr_ESF[:, 0] == order_and_freq[0])
                    self.index_lib = np.delete(self.index_lib, row_index[0], 0)
                    self.Freq_Dr_ESF = np.delete(self.Freq_Dr_ESF, row_index[0], 0)
                    self.ax.scatter(order_and_freq[0], temp_order, color="#0000FF")
                    self.fig.canvas.draw()



        if event.button == 2:
            self.Freq_Dr_ESF = np.delete(self.Freq_Dr_ESF, 0, 0)
            np.savetxt('Freq_Dr_ESF.dat', self.Freq_Dr_ESF)
            event.canvas.mpl_disconnect(self.cid)

        return self
