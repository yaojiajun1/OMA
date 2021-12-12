# _*_ coding:UTF-8 _*_
# developer: yaoji
# Time: 11/12/20204:18 PM

import numpy as np
from builtins import object
from scipy import signal
from scipy import spatial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Model(object):

    def __init__(
        self,
        Tastfrequenz=2000,
        ordnung=20,
        reference=1,
        Daten=[]
    ):
        self.Tastfrequenz = Tastfrequenz
        self.ordnung = ordnung
        self.reference = reference
        self.Daten = Daten

    def SVD_raw_Daten(self):
        Tastfrequenz = self.Tastfrequenz
        Daten = self.Daten
        channel = Daten.shape[0]

        psd = np.zeros((Tastfrequenz // 2 + 1) * Daten.shape[0] * Daten.shape[0], dtype=complex).reshape(
            Tastfrequenz // 2 + 1, Daten.shape[0], Daten.shape[0])
        S = np.zeros(Tastfrequenz // 2 + 1)
        for i in range(channel):
            for j in range(channel):
                f, Pxy = signal.coherence(Daten[i], Daten[j], Tastfrequenz, nperseg=Tastfrequenz)
                psd[:, i, j] = np.abs(Pxy)

        for i in range(psd.shape[0]):
            u, s, v = np.linalg.svd(psd[i, :, :])
            S[i] = 20 * np.log10(np.abs(s[0]))

        return f, S

    def R_dach(self, Daten, ordnung):
        c = Daten.shape[0]
        N = Daten.shape[1]
        set = 2 * ordnung
        size = c * set * c

        R_dach = np.zeros(size).reshape(c, size // c)
        arange = np.arange(1, 2 * ordnung+1, 1)

        for i in arange:
            R_dach[:, (set-i) * c:(set-i+1)*c] = 1 / (N - 1) * Daten[:, :N - i].dot(Daten[:, i:N].T)

        return R_dach, c


    def Toeplitz_matrix(self, R_dach, c, lag, ordnung):

        size = c**2 * ordnung**2            # Matrix size
        set = 2 * ordnung                # R_dach hat x set von c*c matrix

        toeplitz_matrix = np.zeros(size).reshape(c*ordnung, c*ordnung)
        arange = np.arange(1, ordnung + 1, 1)

        for i in arange:
            toeplitz_matrix[(i-1)*c:i*c,:] = R_dach[:, (set-i+1-ordnung-lag)*c:(set-i+1-lag)*c]
        return toeplitz_matrix



    """
    Here (check already)
    """
    def cov_SSI(self):
        ordnung = self.ordnung
        Tastfrequenz = self.Tastfrequenz
        Daten = self.Daten
        reference = self.reference

        Nodes = Daten.shape[0]
        Psi_m_matrix = np.zeros(Nodes ** 2 * ordnung ** 2, dtype=complex).reshape(Nodes * ordnung, Nodes * ordnung)
        Frequenz_natural = np.zeros(ordnung ** 2).reshape(ordnung, ordnung)
        Frequenz_damped = np.zeros(ordnung ** 2).reshape(ordnung, ordnung)
        Damping_ratio = np.zeros(ordnung ** 2).reshape(ordnung, ordnung)


        arange = np.arange(1, ordnung + 1, 1)
        for i in arange:
            model = Model(Tastfrequenz=2000)
            R_dach, c = model.R_dach(Daten, i)  # (check)
            toeplitz_matrix_1 = model.Toeplitz_matrix(R_dach, c, lag=0, ordnung=i)  # Toeplitz matrix T1 (check)
            u, s, v = np.linalg.svd(toeplitz_matrix_1)
            toeplitz_matrix_2 = model.Toeplitz_matrix(R_dach, c, lag=1, ordnung=i)  # Toeplitz matrix T2 (check)

            Matrix_O_plus = np.sqrt(np.diagflat(s ** -1)).dot(u.T)  # (check)
            Matrix_Tau_plus = (v.T).dot(np.sqrt(np.diagflat(s ** -1)))
            Matrix_C = u.dot(np.sqrt(np.diagflat(s)))[0:Nodes, :]  # Matrix C (check)

            Matrix_A = np.linalg.multi_dot([Matrix_O_plus, toeplitz_matrix_2, Matrix_Tau_plus])  # Matrix A
            eigenwert_A, eigenvector_A = np.linalg.eig(Matrix_A)

            arange = np.arange(1, i + 1, 1)
            for j in arange:
                Psi_m = Matrix_C.dot(eigenvector_A[:, (j - 1) * Nodes: j * Nodes])
                Psi_m_matrix[(i - 1) * Nodes:i * Nodes, (j - 1) * Nodes: j * Nodes] = Psi_m

            Lambda = np.zeros(i, dtype=complex)
            for j in arange:
                Lambda[j - 1] = np.log([eigenwert_A[(j - 1) * Nodes + reference - 1]]) * Tastfrequenz

            frequenz_natural = np.zeros(i)
            for j in arange:
                frequenz_natural[j - 1] = np.abs([Lambda[j - 1] / 2 / np.pi])

            frequenz_damped = np.zeros(i)
            for j in arange:
                frequenz_damped[j - 1] = np.imag([Lambda[j - 1] / 2 / np.pi])

            damping_ratio = np.zeros(i)
            for j in arange:
                damping_ratio[j - 1] = -np.real(Lambda[j - 1] / np.abs([Lambda[j - 1]]))

            Frequenz_natural[i - 1, 0:i] = frequenz_natural  # cache
            Frequenz_damped[i - 1, 0:i] = frequenz_damped  # cache
            Damping_ratio[i - 1, 0:i] = damping_ratio  # cache

        Psi_m_Matrix = Psi_m_matrix[:, reference-1::5]
        cache = {'Frequenz_natural': Frequenz_natural, 'Frequenz_damped':Frequenz_damped, 'Damping_ratio':Damping_ratio,
                 'Psi_m_Matrix': Psi_m_Matrix}
        return cache


    def Stablisation_Diagramm(self, cache):
        ordnung = self.ordnung
        Daten = self.Daten
        reference = self.reference


        Frequenz_natural = cache['Frequenz_natural']
        Frequenz_damped = cache['Frequenz_damped']
        Damping_ratio = cache['Damping_ratio']
        Psi_m_Matrix = cache['Psi_m_Matrix']

        Frequenz_stability = np.zeros(ordnung * ordnung).reshape(ordnung, ordnung)
        Damping_ratio_stability = np.zeros(ordnung * ordnung).reshape(ordnung, ordnung)
        MAC_stability = np.zeros(ordnung * ordnung).reshape(ordnung, ordnung)
        # Frequency compare     (check)
        arange = np.arange(1, ordnung, 1)
        for ordnung_zaehler in arange:
            for element_zaehler in range(ordnung_zaehler):
                A = np.abs((Frequenz_natural[ordnung_zaehler - 1, :ordnung_zaehler] -
                            Frequenz_natural[ordnung_zaehler, element_zaehler]))
                A = np.where(A > 5, 2000, A)                                   # Begrenzen die Toleranz bis zu 5 Hz
                Frequency_stable = np.divide(A, Frequenz_natural[ordnung_zaehler - 1, :ordnung_zaehler])  # (check)

                if any(y < 0.01 for y in Frequency_stable):
                    Frequenz_stability[ordnung_zaehler, element_zaehler] = 1
                    index_freq = np.where(Frequency_stable < 0.01)
                    # damping
                    for damping_index in index_freq:
                        B = np.abs(Damping_ratio[ordnung_zaehler - 1, damping_index] - Damping_ratio[
                            ordnung_zaehler, element_zaehler])
                        Damping_stable = np.divide(B, Damping_ratio[ordnung_zaehler - 1, damping_index])

                        if np.any(Damping_stable < 0.05) and np.any(
                                Damping_ratio[ordnung_zaehler, element_zaehler] > 0):
                            Damping_ratio_stability[ordnung_zaehler, element_zaehler] = 1  # (check)

                            Psi_low = Psi_m_Matrix[
                                      (ordnung_zaehler - 1) * Daten.shape[0]:ordnung_zaehler * Daten.shape[0],
                                      damping_index]
                            Psi_high = Psi_m_Matrix[
                                       ordnung_zaehler * Daten.shape[0]:(ordnung_zaehler + 1) * Daten.shape[0],
                                       element_zaehler]
                            numerator = np.abs((np.conj(Psi_low).T).dot(Psi_high)) ** 2
                            denominator = (np.conj(Psi_low).T).dot(Psi_low) * (np.conj(Psi_high).T).dot(Psi_high)
                            mac_stable = np.divide(numerator, denominator)
                            if np.any(1 - mac_stable < 0.02):
                                MAC_stability[ordnung_zaehler, element_zaehler] = 1

        stable_matrix = np.zeros(ordnung * ordnung).reshape(ordnung, ordnung)
        stable_matrix += Frequenz_natural * MAC_stability

        return stable_matrix

    def get_ESF_and_damping_ratio(self, number_of_points, points, cache):
        Daten = self.Daten

        index_location = np.zeros(number_of_points*2).reshape((number_of_points, 2))
        Frequenz_natural = cache['Frequenz_natural']
        Damping_ratio = cache['Damping_ratio']
        Psi_m_Matrix = cache['Psi_m_Matrix']

        Freq_Dr_ESF = np.zeros(number_of_points * (Daten.shape[0] + 2)).reshape(
            (number_of_points, (Daten.shape[0] + 2)))

        for i in range(number_of_points):
            index = plt.ginput()
            index_location[i, :] = index[0]
            ordnung_and_freq = points[spatial.KDTree(points).query(index_location[i, :])[1]]
            temp_ordnung = int(ordnung_and_freq[1])
            temp_index = int((np.where(Frequenz_natural[temp_ordnung, :] == ordnung_and_freq[0]))[0])
            picked_dr = Damping_ratio[temp_ordnung, temp_index]
            picked_esf = np.real(
                Psi_m_Matrix[temp_ordnung * Daten.shape[0]:temp_ordnung * Daten.shape[0] + 5, temp_index])
            Freq_Dr_ESF[i, 0] = ordnung_and_freq[0]
            Freq_Dr_ESF[i, 1] = picked_dr
            Freq_Dr_ESF[i, 2::] = picked_esf

        return Freq_Dr_ESF

    def plot_Eigenschwingform(self, Freq_Dr_ESF, messung_position, number_of_points):
        fig, axs = plt.subplots(number_of_points, 1, constrained_layout=True)
        for i in range(number_of_points):
            Freq_Dr_ESF[i, 2::] = Freq_Dr_ESF[i, 2::]/np.max(Freq_Dr_ESF[i, 2::])# Set the max move to 1
            axs[i].plot(messung_position, np.array(Freq_Dr_ESF[i, 2::]))
            f2 = interp1d(messung_position, np.array(Freq_Dr_ESF[i, 2::]), kind='quadratic')
            xnew = np.linspace(messung_position[0], messung_position[4], num=201, endpoint=True)
            axs[i].plot(xnew, f2(xnew), '--')
            axs[i].set_title('mode %i' % (i + 1) + '   Natural Frequenz: %1.11f' % Freq_Dr_ESF[i, 0] + "  Damping Ratio: %1.11f" % Freq_Dr_ESF[i, 1])
            axs[i].set_xlabel('distance (cm)')
            # axs[i].set_ylabel('Verschiebung')
        plt.show()

    def DD_SSI(self):
        ordnung = self.ordnung
        Tastfrequenz = self.Tastfrequenz
        Daten = self.Daten
        reference = self.reference

        Nodes = Daten.shape[0]
        Psi_m_matrix = np.zeros(Nodes ** 2 * ordnung ** 2, dtype=complex).reshape(Nodes * ordnung, Nodes * ordnung)
        Frequenz_natural = np.zeros(ordnung ** 2).reshape(ordnung, ordnung)
        Frequenz_damped = np.zeros(ordnung ** 2).reshape(ordnung, ordnung)
        Damping_ratio = np.zeros(ordnung ** 2).reshape(ordnung, ordnung)

        channel = Daten.shape[0]

        arange = np.arange(1, ordnung + 1, 1)
        for i in arange:
            length = Daten.shape[1] - 2 * ordnung
            size = channel * ordnung * length * 2
            Block_Hankel_matrix = np.zeros(size).reshape(channel * ordnung*2, length)
            arange_sub = np.arange(1, ordnung * 2 + 1, 1)

            for j in arange_sub:
                Block_Hankel_matrix[(j-1)*channel:j*channel] = Daten[:, j-1:j-1+length]

            Block_Hankel_matrix *= 1 / np.sqrt(length)
            # ==================================================
            # compute Projection
            Y_p = Block_Hankel_matrix[0:channel * ordnung, :]
            Y_f = Block_Hankel_matrix[channel * ordnung::, :]
            Matrix_Y_i_i = Block_Hankel_matrix[channel * ordnung:channel * (ordnung + 1), :]
            Y_p_plus = Block_Hankel_matrix[0:channel * (ordnung + 1), :]
            Y_f_minus = Block_Hankel_matrix[channel * (ordnung + 1)::, :]

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
            Kalman_filter_state_i_plus_1 = np.linalg.pinv(Observation_Matrix_i_upper).dot(Projection_i_minus_1)

            # ==================================================
            # Use the First Method
            Matrix_AC_1 = np.vstack([Kalman_filter_state_i_plus_1, Matrix_Y_i_i])
            Matrix_AC = Matrix_AC_1.dot(np.linalg.pinv(Kalman_filter_state_i))
            Matrix_A = Matrix_AC[0:-1 * channel, :]
            Matrix_C = Matrix_AC[-1 * channel::, :]

            eigenwert_A, eigenvector_A = np.linalg.eig(Matrix_A)
            # ==================================================

            arange = np.arange(1, i + 1, 1)
            for j in arange:
                Psi_m = Matrix_C.dot(eigenvector_A[:, (j - 1) * Nodes: j * Nodes])
                Psi_m_matrix[(i - 1) * Nodes:i * Nodes, (j - 1) * Nodes: j * Nodes] = Psi_m

            Lambda = np.zeros(i, dtype=complex)
            for j in arange:
                Lambda[j - 1] = np.log([eigenwert_A[(j - 1) * Nodes + reference - 1]]) * Tastfrequenz

            frequenz_natural = np.zeros(i)
            for j in arange:
                frequenz_natural[j - 1] = np.abs([Lambda[j - 1] / 2 / np.pi])

            frequenz_damped = np.zeros(i)
            for j in arange:
                frequenz_damped[j - 1] = np.imag([Lambda[j - 1] / 2 / np.pi])

            damping_ratio = np.zeros(i)
            for j in arange:
                damping_ratio[j - 1] = -np.real(Lambda[j - 1] / np.abs([Lambda[j - 1]]))

            Frequenz_natural[i - 1, 0:i] = frequenz_natural  # cache
            Frequenz_damped[i - 1, 0:i] = frequenz_damped  # cache
            Damping_ratio[i - 1, 0:i] = damping_ratio  # cache

        Psi_m_Matrix = Psi_m_matrix[:, reference - 1::5]
        cache = {'Frequenz_natural': Frequenz_natural, 'Frequenz_damped': Frequenz_damped,
                 'Damping_ratio': Damping_ratio,
                 'Psi_m_Matrix': Psi_m_Matrix}
        return cache