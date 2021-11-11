"""
Utilities for loading data from mat file.

Author: Christine Evers, c.evers@imperial.ac.uk
"""
import scipy.io as sio
import pdb
import numpy as np
import tensorflow as tf
import os

def mat_to_tfrecord(mat_dir_feat_train):
    """
    Recursively reads a directory of mat files into one tfrecord

    Input:
    - Directory containing one mat file for each scenario. The following folder structure is required:

    Output:
    - $save_dir/test.tfrecord
    - $save_dir/train.tfrecord
    """

    def get_subdirs(a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

    # Pre-allocate table, ready for data:
    subdirs_pre = get_subdirs(mat_dir_feat_train)


    subdirs = [subdir for subdir in subdirs_pre if int(subdir) < len(subdirs_pre)*(2/3)]
    # subdirs_data= {
    #     'subdirs': subdirs
    # }
    # sio.savemat(f'{save_dir}subdirs_data.mat',
    #             subdirs_data)
    mat_contents_val_idx = sio.loadmat('/home/al5517/ML_data/directional_2021_06_02/labels_val_recon.mat')
    val_idx = mat_contents_val_idx['labels_val_recon']
    val_idx = val_idx.flatten()
    val_idx = val_idx - 1

    tfrecord_dir = '/home/al5517/ML_data/directional_2021_06_02/tfrecord/stftmag_gd/'
    if (not (os.path.isdir(tfrecord_dir))):
        os.makedirs(tfrecord_dir)
    # split into test dataset and train+val dataset - Ensures scenarios used during testing are unseen during training / validation:
    #max_idx_value = int(len(subdirs) / 2 - 1)  # normally this would just be len(subdirs)
    #val_idx_other = np.random.choice(len(subdirs), int(np.around(len(subdirs)*val_size)), replace=False)
    # on second round of training here, val_idx = val_idx + len(subdirs)/2
    #print(f'new val_idx: {val_idx}')
    #print(f'old val_idx: {val_idx_other}')
    val_subdirs = np.array(subdirs)[val_idx]
    #print(f'val_subdirs: {val_subdirs}')
    remaining_subdirs = np.delete(subdirs, val_idx)

    # Write data in test + train dataset to tfrecord:
    print('Saving val.tfrecord')
    total_sentences_val, nfeatures, nb_frames, X_val, y_val = nparray_to_tfrecord(mat_dir_feat_train, val_subdirs, tfrecord_dir+'val.tfrecord')
    print('Saving train.tfrecord')
    total_sentences_train, _, _, _, _ = nparray_to_tfrecord(mat_dir_feat_train, remaining_subdirs, tfrecord_dir+'train.tfrecord')

    return total_sentences_train, total_sentences_val, nfeatures, nb_frames, X_val, y_val


def mat_to_nparray_test(mat_dir_feat_test):
    """
    Recursively reads a directory of mat files into one tfrecord

    Input:
    - Directory containing one mat file for each scenario. The following folder structure is required:

    Output:
    - $save_dir/test.tfrecord
    - $save_dir/train.tfrecord
    """

    def get_subdirs(a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

    # Pre-allocate table, ready for data:
    test_subdirs = get_subdirs(mat_dir_feat_test)

    # split into test dataset and train+val dataset - Ensures scenarios used during testing are unseen during training / validation:
    #test_idx = np.random.choice(len(subdirs), int(np.around(len(subdirs)*test_size)), replace=False)
    #test_subdirs = np.array(subdirs)[test_idx]
    #remaining_subdirs = np.delete(subdirs, test_idx)
    tfrecord_dir = '/home/al5517/ML_data/directional_2021_06_02/tfrecord/stftmag_gd/'
    if (not (os.path.isdir(tfrecord_dir))):
        os.makedirs(tfrecord_dir)

    #subdirs_test = get_subdirs(mat_dir_test)
    # Write data in test + train dataset to tfrecord:
    print('Saving test.tfrecord')
    total_sentences_test, nfeatures, nb_frames, X_test, y_test = nparray_to_tfrecord(mat_dir_feat_test, test_subdirs, tfrecord_dir+'test.tfrecord')

    return total_sentences_test, total_sentences_test, nfeatures, nb_frames, X_test, y_test




def nparray_to_tfrecord(mat_dir_feat, subdirs, tfrecrod_fname):
    """
    Loads mat file to nparray and writes the array to tfrecord (standard format for datasets in tensorflow)
    """

    def load_mat_features(fname_mag, fname_phase, fname_H):
        """Load data and labels from a single mat file"""

        # print(sio.whosmat(fname))
        mat_contents_mag = sio.loadmat(fname_mag)
        mat_contents_phase = sio.loadmat(fname_phase)
        mat_contents_H = sio.loadmat(fname_H)
        cl_mag = mat_contents_mag['xl_hat_mag']
        cl_phase = mat_contents_phase['xl_hat_angle']
        cr_mag = mat_contents_mag['xr_hat_mag']
        cr_phase = mat_contents_phase['xr_hat_angle']
        dl_mag = mat_contents_mag['yl_hat_mag']
        dl_phase = mat_contents_phase['yl_hat_angle']
        dr_mag = mat_contents_mag['yr_hat_mag']
        dr_phase = mat_contents_phase['yr_hat_angle']
        H = mat_contents_H['H_trunc']


        for i in range(0,cl_phase.shape[1]-1):
            cl_phase[:,cl_phase.shape[1] - i] = cl_phase[:,cl_phase.shape[1] - i] - cl_phase[:,cl_phase.shape[1] - (i+1)]
            cr_phase[:, cr_phase.shape[1] - i] = cr_phase[:, cr_phase.shape[1] - i] - cr_phase[:,
                                                                                      cr_phase.shape[1] - (i + 1)]
            dl_phase[:, dl_phase.shape[1] - i] = dl_phase[:, dl_phase.shape[1] - i] - dl_phase[:,
                                                                                      dl_phase.shape[1] - (i + 1)]
            dr_phase[:, dr_phase.shape[1] - i] = dr_phase[:, dr_phase.shape[1] - i] - dr_phase[:,
                                                                                      dr_phase.shape[1] - (i + 1)]


        C = np.empty((2 * cl_mag.shape[0] + 2 * cl_phase.shape[0], cl_mag.shape[1]), dtype=cl_mag.dtype)
        C[0:(cl_mag.shape[0]), :] = cl_mag
        C[(cl_mag.shape[0]):(2*cl_mag.shape[0]),:] = cr_mag
        C[(2*cl_mag.shape[0]):(2*cl_mag.shape[0] + cl_phase.shape[0]), :] = cl_phase
        C[(2*cl_mag.shape[0] + cl_phase.shape[0]):(2*cl_mag.shape[0] + 2*cl_phase.shape[0]), :] = cr_phase

        # print(f'Clean features dim {C.shape}')
        dim_max = 483
        C_zp = np.zeros((C.shape[0], dim_max))
        C_zp[:C.shape[0], :C.shape[1]] = C
        # X = np.concatenate(xl,xr,axis=0)
        # print(f'Clean features dim zp {C_zp.shape}')

        D = np.empty((2*dl_mag.shape[0] + 2*dl_phase.shape[0], dl_mag.shape[1]), dtype=dl_mag.dtype)

        D[0:(dl_mag.shape[0]), :] = dl_mag
        D[(dl_mag.shape[0]):(2*dl_mag.shape[0]),:] = dr_mag
        D[(2*dl_mag.shape[0]):(2*dl_mag.shape[0] + dl_phase.shape[0]), :] = dl_phase
        D[(2*dl_mag.shape[0] + dl_phase.shape[0]):(2*dl_mag.shape[0] + 2*dl_phase.shape[0]), :] = dr_phase

        D_zp = np.zeros((D.shape[0], dim_max))
        D_zp[:D.shape[0], :D.shape[1]] = D
        # Y = np.concatenate(yl,yr,axis=0)
        #print(f'Distorted features dim zp {D_zp.shape}')

        X = np.empty((C_zp.shape[0], C_zp.shape[1],2), dtype=C_zp.dtype)

        #C_zp = np.expand_dims(C_zp, axis = 2)
        #D_zp = np.expand_dims(D_zp, axis = 2)
        X[:,:,0] = C_zp
        X[:,:,1] = D_zp

        # X = mat_contents['features']
        # nframes, nfeatures, nfreqs = X.shape
        # nlabels = mat_contents['nLabels'][0][0]
        # y = np.array(mat_contents['labels'].todense())
        # y = np.reshape(y, (nframes,nfreqs,nlabels))
        # y = np.transpose(y, (0,2,1))

        # Separate magnitude and phase:
        # X_mag = np.absolute(X)
        # X_phase = np.angle(X)
        # X = np.append(X_mag, X_phase, axis=1)
        # nfeatures *= 2

        return X
    def load_mat_label(fname):
        """Load data and labels from a single mat file"""

        # print(sio.whosmat(fname))
        mat_contents = sio.loadmat(fname)
        sip_array = mat_contents['mbstoi']
        sip = np.mean(sip_array, axis=None, dtype=None)
        #print(f'Output MBSTOI {sip}')
        #print(f'Output MBSTOI data type {sip.dtype}')
        #print(f'Output MBSTOI value shape {sip.shape}')
        #sip = sip[0]
        #print(f'New output MBSTOI value shape {sip.shape}')
        #nframes, nfeatures, nfreqs = X.shape
        #nlabels = mat_contents['nLabels'][0][0]
        #y = np.array(mat_contents['labels'].todense())
        #y = np.reshape(y, (nframes,nfreqs,nlabels))
        #y = np.transpose(y, (0,2,1))

        # Separate magnitude and phase:
        #X_mag = np.absolute(X)
        #X_phase = np.angle(X)
        #X = np.append(X_mag, X_phase, axis=1)
        #nfeatures *= 2

        return sip

    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    # Read data in remaining subdirs, reserved for train + val data:
    #total_frames = 0
    mat_fname_features_mag = mat_dir_feat + subdirs[0] + '/stft_mag.mat'
    mat_fname_features_phase = mat_dir_feat + subdirs[0] + '/stft_phase.mat'
    mat_fname_H = mat_dir_feat + 'H_trunc.mat'
    X = load_mat_features(mat_fname_features_mag, mat_fname_features_phase, mat_fname_H)

    total_sentences = len(subdirs)
    nfeatures = X.shape[0]
    nb_frames = X.shape[1]
    X_all =np.zeros((total_sentences, nfeatures, nb_frames, X.shape[2]))
    sip_all = np.zeros((total_sentences))
    for idx in range(len(subdirs)):
        mat_fname_features_mag = mat_dir_feat + subdirs[idx] + '/stft_mag.mat'
        mat_fname_features_phase = mat_dir_feat + subdirs[idx] + '/stft_phase.mat'
        X = load_mat_features(mat_fname_features_mag, mat_fname_features_phase, mat_fname_H)

        mat_fname_label = mat_dir_feat + subdirs[idx] + '/mbstoi.mat'
        sip = load_mat_label(mat_fname_label)

        X_all[idx, :, :, :] = X
        sip_all[idx] = sip

    # Write data to tfrecord - Standard format for datasets in tensorflow
    with tf.io.TFRecordWriter(str(tfrecrod_fname)) as writer:
        for row_idx in range(X.shape[0]):
            # Read features and flatten matrices, ready for tfrecord:
            X_row = X_all[row_idx,:,:,:].reshape(-1)
            y_row = sip_all[row_idx].reshape(-1)

            # Write tf feature to tfrecord:
            feature = {
                'data': float_feature(X_row),
                'label': float_feature(y_row),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            example = example_proto.SerializeToString()
            writer.write(example)

    return total_sentences, nfeatures, nb_frames, X_all, sip_all

def tfrecord_to_dataset(nfeatures, nb_frames, tfrecord_fname, batch_size, shuffle_buffer):
    """
    Converts tfrecord file to data generator

    Make sure that the batches / frames are in the first dimension of the tensor you create!
    """

    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = { 'data': tf.io.FixedLenFeature([nfeatures,nb_frames,2], tf.float32),\
                             'label': tf.io.FixedLenFeature([1], tf.float32), }

        # Load one example
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        #bob = 1
        # Turn your saved string into an array
        parsed_features['data'] = tf.reshape( tf.reshape( parsed_features['data'], (1,nfeatures,nb_frames,2))[:,:,:,:], (nfeatures,nb_frames,2))
        parsed_features['label'] = tf.reshape( tf.reshape( tf.cast( parsed_features['label'], tf.float64), (1,1) )[:], (1,))

        return parsed_features['data'], parsed_features['label']

    # This works with arrays as well
    tfrecord_dir = '/home/al5517/ML_data/directional_2021_06_02/tfrecord/stftmag_gd/'
    dataset = tf.data.TFRecordDataset(tfrecord_dir + tfrecord_fname)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever for validation and training data - not required for test data.
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(shuffle_buffer)

    # Set the batchsize
    dataset = dataset.batch(batch_size)

    return dataset