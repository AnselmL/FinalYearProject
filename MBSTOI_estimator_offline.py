import nnresample
import resampy
from scipy import signal
import scipy.io as sio
from scipy.io import wavfile
from MBSTOI_estimate_utils import *
import tensorflow as tf



clean_audio_in_dir = 'D:/FYP/feature_preprocessing_input/directional_2021_06_02/TRAIN/1/clean.wav'
degraded_audio_in_dir = 'D:/FYP/feature_preprocessing_input/directional_2021_06_02/TRAIN/1/mixed.wav'

fs_preprocessing = 10000
range = 40
N = 256
K = 256


#read audio
fs_clean, x = wavfile.read(clean_audio_in_dir)
fs_degraded, y = wavfile.read(degraded_audio_in_dir)


#zero-mean unit variance normalisation
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

#resampling

if fs_clean != fs_preprocessing:
    x = resampy.resample(x, fs_clean, fs_preprocessing, filter='kaiser_fast')
    #x = signal.resample(x, int((fs_preprocessing/fs_clean)*x.shape[0]), )
    #x = nnresample.resample(x, fs_preprocessing, fs_clean, axis=0)

if fs_degraded != fs_preprocessing:
    #y = nnresample.resample(y, fs_preprocessing, fs_degraded, axis=0)
    #y = signal.resample(y, int((fs_preprocessing/fs_degraded)*y.shape[0]))
    y = resampy.resample(y, fs_degraded, fs_preprocessing, filter='kaiser_fast')

if x.ndim == 1:
    xl = x[:]
    xr = x[:]
elif x.ndim == 2:
    xl = x[:,0]
    xr = x[:,1]

yl = y[:,0]
yr = y[:,1]

xl, xr, yl, yr = remove_silent_frames(xl, xr, yl, yr, range, N, K)

xl_hat = stdft(xl, N, N/2, K)
xr_hat = stdft(xr, N, N/2, K)
yl_hat = stdft(yl, N, N/2, K)
yr_hat = stdft(yr, N, N/2, K)

xl_hat_mag = np.real(xl_hat)
xr_hat_mag = np.real(xr_hat)

yl_hat_mag = np.real(xl_hat)
yr_hat_mag = np.real(xr_hat)

xl_hat_angle = np.real(xl_hat)
xr_hat_angle = np.real(xr_hat)

yl_hat_angle = np.real(xl_hat)
yr_hat_angle = np.real(xr_hat)

featDim = 483
C = np.zeros((2*xl_hat_mag.shape[0] + 2*xl_hat_angle.shape[0],xl_hat_mag.shape[0]))
C[0::4] = xl_hat_mag
C[1::4] = xl_hat_angle
C[2::4] = xr_hat_mag
C[3::4] = xr_hat_angle

D = np.zeros((2*yl_hat_mag.shape[0] + 2*yl_hat_angle.shape[0],yl_hat_mag.shape[0]))
D[0::4] = yl_hat_mag
D[1::4] = yl_hat_angle
D[2::4] = yr_hat_mag
D[3::4] = yr_hat_angle

C_zp = np.zeros((C.shape[0], featDim))
C_zp[:C.shape[0], :C.shape[1]] = C

D_zp = np.zeros((D.shape[0], featDim))
D_zp[:D.shape[0], :D.shape[1]] = D


X = np.empty((C_zp.shape[0], C_zp.shape[1], 2), dtype=C_zp.dtype)

# C_zp = np.expand_dims(C_zp, axis = 2)
# D_zp = np.expand_dims(D_zp, axis = 2)
X[:, :, 0] = C_zp
X[:, :, 1] = D_zp

nn_model = tf.keras.models.load_model('/my_model/')

MBSTOI_estimate = nn_model.predict((X))


#do pre-processing


#input to neural network as predict

#