import numpy as np


def remove_silent_frames(xl, xr, yl, yr, dyn_range, N, K):

    xl = xl.flatten()
    xr = xr.flatten()
    yl = yl.flatten()
    yr = yr.flatten()

    frames = range(0,xl.shape[0]-N,K)
    print(f'length frames: {len(frames)}')
    w = np.hanning(N)
    print(f'w diemension: {w.shape}')
    msk_l = np.zeros((len(frames),))
    msk_r = np.zeros((len(frames),))

    for j in range(0,len(frames)):
        jj = range(frames[j],(frames[j]+N))
        msk_l[j] = 20 * np.log10(np.divide(np.linalg.norm(np.multiply(xl[jj], w), 2),np.sqrt(N)))
        msk_r[j] = 20 * np.log10(np.divide(np.linalg.norm(np.multiply(xr[jj], w), 2), np.sqrt(N)))

    msk_l = (msk_l - np.amax(np.concatenate((msk_l, msk_r))+ dyn_range)) > 0
    msk_r = (msk_r - np.amax(np.concatenate((msk_l, msk_r)) + dyn_range)) > 0
    msk = np.any(np.concatenate((msk_l,msk_r)),axis=0)
    count = 0

    xl_sil = np.zeros((xl.shape[0],xl.shape[1]))
    xr_sil = np.zeros((xr.shape[0],xr.shape[1]))
    yl_sil = np.zeros((yl.shape[0],yl.shape[1]))
    yr_sil = np.zeros((yr.shape[0],yr.shape[1]))

    for j in range(0, len(frames)):
        if msk(j):
            jj_i = range(frames[j],(frames[j]+N))
            jj_o = range(frames[count],frames[count]+N)
            xl_sil[jj_o] = xl_sil[jj_o] + np.multiply(xl[jj_i],w)
            xr_sil[jj_o] = xr_sil[jj_o] + np.multiply(xr[jj_i], w)
            yl_sil[jj_o] = yl_sil[jj_o] + np.multiply(yl[jj_i],w)
            yr_sil[jj_o] = yr_sil[jj_o] + np.multiply(yr[jj_i], w)
            count += 1

    xl_sil = xl_sil[0:jj_o[-1]]
    xr_sil = xr_sil[0:jj_o[-1]]
    yl_sil = yl_sil[0:jj_o[-1]]
    yr_sil = yr_sil[0:jj_o[-1]]

    return xl_sil, xr_sil, yl_sil, yr_sil

def stdft(x, N, K, N_fft):
    frames = range(0,x.shape[0]-N,K)
    x_stdft = np.zeros((len(frames),N_fft))
    w = np.hanning(N)
    x = x.flatten

    for i in range(0,len(frames)):
        ii = range(frames[i], frames[i]+N)
        x_stdft[i, :] = np.fft.fft(np.multiply(x[ii], w), N_fft)

