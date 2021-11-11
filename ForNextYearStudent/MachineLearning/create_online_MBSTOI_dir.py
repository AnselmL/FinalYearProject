import scipy.io as sio
import os
import numpy as np
base_directory = '/home/al5517/feature_preprocessing_output/directional_2021_06_02/'
out_base_dir = '/home/al5517/feature_preprocessing_output/directional_online_redo_2021_06_02/'
extensions = ['TRAIN','TEST']


def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

for ext in extensions:
    count = 1
    subdirs = get_subdirs(f'{base_directory}{ext}/')
    for subdir in subdirs:
        #load mat file

        mat_contents_mag = sio.loadmat(f'{base_directory}{ext}/{subdir}/stft_mag.mat')
        mat_contents_phase = sio.loadmat(f'{base_directory}{ext}/{subdir}/stft_phase.mat')
        mat_contents_mbstoi = sio.loadmat(f'{base_directory}{ext}/{subdir}/mbstoi.mat')
        xl_hat_mag_full = mat_contents_mag['xl_hat_mag']
        xl_hat_angle_full = mat_contents_phase['xl_hat_angle']
        xr_hat_mag_full = mat_contents_mag['xr_hat_mag']
        xr_hat_angle_full = mat_contents_phase['xr_hat_angle']
        yl_hat_mag_full = mat_contents_mag['yl_hat_mag']
        yl_hat_angle_full = mat_contents_phase['yl_hat_angle']
        yr_hat_mag_full = mat_contents_mag['yr_hat_mag']
        yr_hat_angle_full = mat_contents_phase['yr_hat_angle']
        mbstoi_full = mat_contents_mbstoi['mbstoi']
        rand_i = np.random.choice(mbstoi_full.shape[1], 4, replace=False)
        #for i in range(0,mbstoi_full.shape[1]):
        for i in rand_i:
            xl_hat_mag = xl_hat_mag_full[:,i:(i+30)]
            xr_hat_mag = xr_hat_mag_full[:, i:(i + 30)]

            xl_hat_angle = xl_hat_angle_full[:,i:(i+30)]
            xr_hat_angle = xr_hat_angle_full[:, i:(i + 30)]

            yl_hat_mag = yl_hat_mag_full[:,i:(i+30)]
            yr_hat_mag = yr_hat_mag_full[:, i:(i + 30)]

            yl_hat_angle = yl_hat_angle_full[:,i:(i+30)]
            yr_hat_angle = yr_hat_angle_full[:, i:(i + 30)]

            mbstoi = mbstoi_full[:,i]

            stft_mag = {
                'xl_hat_mag': xl_hat_mag,
                'xr_hat_mag': xr_hat_mag,
                'yl_hat_mag': yl_hat_mag,
                'yr_hat_mag': yr_hat_mag
            }
            stft_phase = {
                'xl_hat_angle': xl_hat_angle,
                'xr_hat_angle': xr_hat_angle,
                'yl_hat_angle': yl_hat_angle,
                'yr_hat_angle': yr_hat_angle
            }
            mbstoi_save = {
                'mbstoi': mbstoi
            }
            if not (os.path.isdir(f'{out_base_dir}/{ext}/{count}/')):
                os.makedirs(f'{out_base_dir}/{ext}/{count}/')
            sio.savemat(f'{out_base_dir}{ext}/{count}/stft_mag.mat',
                        stft_mag)
            sio.savemat(f'{out_base_dir}{ext}/{count}/stft_phase.mat',
                        stft_phase)
            sio.savemat(f'{out_base_dir}{ext}/{count}/mbstoi.mat',
                        mbstoi_save)
            count = count + 1
