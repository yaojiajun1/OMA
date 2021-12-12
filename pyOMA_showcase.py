# _*_ coding:UTF-8 _*_
# developer: yaoji
# Time: 1/29/202112:44 AM
import pyOMA
from scipy.io import loadmat
import numpy as np
# =================================================
# Load daten: .mat as example
# =================================================
X = loadmat("daten/Run_1_z.mat")
Daten = X["Run_1_z"]



# =================================================
############### Initialisation ####################
# =================================================
# use pyOMA.Model to define the initial Condition
# (example for SSI)
# =================================================
model = pyOMA.Model(sampling_frequency=12800, Order=20, Reference=3, Data=Daten, Method='cov_SSI')
messung_position = np.array([3.1, 12.5, 25, 37.5, 47.5])


# =================================================
############ Draw Stabilisation Diagramm ##########
# =================================================
# use model.Stabilisation_Diagramm to plot SD:
# use svd as indicator with (plot_svd=True)
# =================================================
fig, ax, stable_points, cache = model.Stablisation_Diagramm(plot_svd=True)


# =================================================
############ Choose stable points from SD  ########
# =================================================
# use pyOMA.choose_points to choose points in SD
# use left click to choose, and right click to cancel.
# when finish, click the middle button.
# The chosen points, with their Frequency, damping ration and mode
# will be save in "Freq_Dr_ESF.dat"
# =================================================

pyOMA.choose_points(points=stable_points, fig=fig, cache=cache, ax=ax)

# =================================================
############ plot the modes #######################
# =================================================
# run the following code when "Freq_Dr_ESF.dat" is ready
# The chosen mode will be plotted
# =================================================
freq_dr_esf = np.loadtxt("Freq_Dr_ESF.dat")
model.plot_Mode(freq_dr_esf=freq_dr_esf, messung_position=messung_position)


