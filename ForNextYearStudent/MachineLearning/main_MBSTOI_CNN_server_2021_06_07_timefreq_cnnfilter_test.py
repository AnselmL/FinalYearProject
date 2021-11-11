
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
#import tensorflow as tf
from dataset_utils_MBSTOI_CNN_stft import *
from build_model_MBSTOI_CNN_features_timefreq_cnntest_2021_06_20 import *
from tensorflow.keras import backend as K
import pdb
import datetime
import os
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io as sio
import tensorflow.compat.v1 as tf
import gc
tf.disable_v2_behavior()


#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


if __name__ == "__main__":
    mat_dir_feat_train = '/home/al5517/feature_preprocessing_output/directional_2021_06_02/TRAIN/'

    mat_dir_feat_test = '/home/al5517/feature_preprocessing_output/directional_2021_06_02/TEST/'
    iteration = 0
    file_iter = 1
    base_dir = '/home/al5517/ML_data/directional_2021_06_02/timefreq_feat/cnnfilter_upd'
    if (not (os.path.isfile(f'{base_dir}{file_iter}/{iteration}/predictions_data.mat'))):
        data_dir = f'{base_dir}{file_iter}/'
        if (not (os.path.isdir(f'{base_dir}{file_iter}'))):
            os.makedirs(f'{base_dir}{file_iter}')

    else:
        while (os.path.isfile(f'{base_dir}{file_iter}/{iteration}/predictions_data.mat')):
            file_iter = file_iter + 1
            data_dir = f'{base_dir}{file_iter}/'

        if (not (os.path.isdir(f'{base_dir}{file_iter}'))):
            os.makedirs(f'{base_dir}{file_iter}')

    cnn_layer_size = [1,3,5,7]


    BATCH_SIZE = 50
    SHUFFLE_BUFFER_SIZE = 5000
    BATCH_SIZE_VAL = 50
    SHUFFLE_BUFFER_SIZE_VAL = 1000
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    LOSS = tf.keras.losses.MeanSquaredError()
    # Metrics: Output metrics evaluated during validation
    METRICS = [tf.keras.metrics.MeanAbsoluteError()]
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    EPOCHS = 5

    total_sentences_test, _, nfeatures, nb_frames, X_test, y_test = mat_to_nparray_test(mat_dir_feat_test)

    test_dataset = tfrecord_to_dataset(nfeatures, nb_frames, 'test.tfrecord', BATCH_SIZE,
                                       SHUFFLE_BUFFER_SIZE)

    cnn_kernel_size = (4,8)
    cnn_stride = (2,4)
    cnn_pool_kernel = (8,8)

    total_sentences_train, total_sentences_val, nfeatures, nb_frames, X_val, y_val = mat_to_tfrecord(
        mat_dir_feat_train)


    train_dataset = tfrecord_to_dataset(nfeatures, nb_frames, 'train.tfrecord', BATCH_SIZE,
                                        SHUFFLE_BUFFER_SIZE)
    val_dataset = tfrecord_to_dataset(nfeatures, nb_frames, 'val.tfrecord', BATCH_SIZE_VAL,
                                       SHUFFLE_BUFFER_SIZE_VAL)

    for cnnlayer in cnn_layer_size:
        model = build_model(OPTIMIZER, LOSS, METRICS, cnnlayer, cnn_kernel_size, cnn_stride, cnn_pool_kernel, nfeatures, nb_frames) #used to be 36, 3,3

        model.summary()
        params = model.count_params()

        hist = model.fit(train_dataset, validation_data=val_dataset, validation_steps=total_sentences_val//BATCH_SIZE_VAL, epochs=EPOCHS, verbose=1, steps_per_epoch= total_sentences_train//BATCH_SIZE)

        train_loss_results = hist.history
        train_mse = train_loss_results['loss']
        train_mae = train_loss_results['mean_absolute_error']
        val_mse = train_loss_results['val_loss']
        val_mae = train_loss_results['val_mean_absolute_error']

        # Test:
        results = model.evaluate(test_dataset, steps=total_sentences_test // BATCH_SIZE)

        test_mse = results[0]
        test_mae = results[1]
        print(f'test_mae: {test_mae}')
        print(f'test_mse: {test_mse}')

        preds_val = np.reshape(model.predict(X_val), (total_sentences_val))
        preds_test = np.reshape(model.predict(X_test), (total_sentences_test))




        results_data = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'predictions_val': preds_val,
            'labels_val': y_val,
            'predictions_test': preds_test,
            'labels_test': y_test,
            'nb_parameters': params
        }
        del results
        del hist
        del model
        del params
        gc.collect()
        if not (os.path.isdir(f'{data_dir}/{iteration}/')):
            os.makedirs(f'{data_dir}/{iteration}/')
        sio.savemat(f'{data_dir}/{iteration}/predictions_data.mat',
                    results_data)
        iteration = iteration + 1

