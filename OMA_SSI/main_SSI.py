# _*_ coding:UTF-8 _*_
# developer: yaoji
# Time: 11/26/202012:44 AM
import pyOMA
from data_utils import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
t = time.time()
# ==================================================

<<<<<<< HEAD
X = load_Daten("daten/Run_1_z.mat")
=======
Daten = "daten/Run_1_z.mat"
X = load_Daten(Daten)
>>>>>>> afee5a601d25136f9a59056ba7f073b74b94387e
Daten = X["Run_1_z"]

Ordnung = 50
Tastfrequenz = 12800
reference = 3
print('In processing......')

model = pyOMA.Model(Tastfrequenz=Tastfrequenz, ordnung=Ordnung, reference=reference, Daten=Daten)
cache = model.cov_SSI()
stable_matrix = model.Stablisation_Diagramm(cache=cache)

print(time.time() - t)

# plot SD
points_x = stable_matrix[stable_matrix > 0]
points_y = np.nonzero(stable_matrix > 0)[0]
points = np.column_stack((points_x, points_y))
fig = plt.figure()
plt.plot(points_x, points_y, '.', label='Stable Points')
fig.suptitle('Stablization Diagramm', fontsize=20)
plt.xlabel('Frequenz (Hz)', fontsize=20)
plt.ylabel('Ordnung', fontsize=20)


# plot SVD
f, S = model.SVD_raw_Daten()
S /= np.max(S) / np.max(points_y) * 2
fig = plt.plot(f, S, label='Mode indicator Function')
legend = plt.legend(loc='lower right', fontsize=15)
plt.grid()
plt.show()


# plot Einschwingform
number_of_points = 3                                    # define how many points to choose
messung_position = np.array([3.1, 12.5, 25, 37.5, 47.5])       # test structure
# messung_position = np.array([0, 10, 50, 90, 100])      # Johannes_
Freq_Dr_ESF = model.get_ESF_and_damping_ratio(number_of_points=number_of_points, points=points, cache=cache)
model.plot_Eigenschwingform(Freq_Dr_ESF=Freq_Dr_ESF, messung_position=messung_position, number_of_points=number_of_points)


print(time.time() - t)