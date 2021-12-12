# _*_ coding:UTF-8 _*_
# developer: yaoji
# Time: 11/12/20204:56 PM

from scipy.io import loadmat
import numpy as np

def load_Daten(ROOT):
    out = loadmat(ROOT)

    return out

