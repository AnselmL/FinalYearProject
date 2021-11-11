
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
#import tensorflow as tf
from dataset_utils_MBSTOI_CNN_stftmag_phase import *
from build_model_MBSTOI_CNN_actcnn_2021_06_08 import *
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
    file_iter = 1
    iteration = 0
    base_dir = '/home/al5517/ML_data/directional_2021_06_02/parameter_testing/act_funccnn_upd'
    if(not(os.path.isfile(f'{base_dir}{file_iter}/{iteration}/predictions_data.mat'))):
        data_dir = f'{base_dir}{file_iter}/'
        if(not(os.path.isdir(f'{base_dir}{file_iter}'))):
            os.makedirs(f'{base_dir}{file_iter}')

    else:
        while(os.path.isfile(f'{base_dir}{file_iter}/{iteration}/predictions_data.mat')):
            file_iter = file_iter + 1
            data_dir = f'{base_dir}{file_iter}/'

        if (not (os.path.isdir(f'{base_dir}{file_iter}'))):
            os.makedirs(f'{base_dir}{file_iter}')
    #file_write_dir = '/home/al5517/ML_data/simulated_2021_04_08/post_fix/CNN/network_data/'
    #mat_dir_feat_test = '/home/al5517/feature_preprocessing_output/1-fold/measured/mfcc_only/TEST/'
    #mat_dir_lbl_test = '/home/al5517/feature_preprocessing_output/1-fold/measured/mbstoi_lbl/TEST/'
    cnn_layer_size = [8]
    #cnn_kernel_size_mod = [(8, 8), (16, 8)]
    #stride_kernel = [(2,2),(4,2),(8,2),(8,4)]
    #max_pool_kernel = [(16,1),(2,2),(4,2),(8,2),(4,4),(8,4)]
    #max_pool_kernel_mod = [(8,4)]
    first_fc_neurons = [128]
    second_fc_neurons = [128]
    #alpha_values = [0, 0.1, 0.2, 0.3, 0.4]
    #alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    act_func = ['tanh','elu','exponential','selu']
    #act_func = ['tanh']
    #act_func = ['leaky0', 'leaky0.1', 'leaky0.2', 'leaky0.3']
    #act_func = ['leaky0.4', 'leaky0.5','leaky0.5','leaky0.6']
    dropout_values = [0.3]
    max_pool_bool = [False]

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

    total_sentences_test, _, nfeatures, nb_frames, X_test, y_test = mat_to_nparray_test(mat_dir_feat_test, data_dir)

    test_dataset = tfrecord_to_dataset(nfeatures, nb_frames, 'test.tfrecord', BATCH_SIZE,
                                       SHUFFLE_BUFFER_SIZE)

    cnn_kernel_size = [(nfeatures, 1)]



    #

    total_sentences_train, total_sentences_val, nfeatures, nb_frames, X_val, y_val = mat_to_tfrecord(
    mat_dir_feat_train, data_dir, VAL_SPLIT)

    train_dataset = tfrecord_to_dataset(nfeatures, nb_frames, 'train.tfrecord', BATCH_SIZE,
                                        SHUFFLE_BUFFER_SIZE)
    val_dataset = tfrecord_to_dataset(nfeatures, nb_frames, 'val.tfrecord', BATCH_SIZE_VAL,
                                       SHUFFLE_BUFFER_SIZE_VAL)
    niterations = len(act_func)


    train_mse = np.zeros((niterations, EPOCHS))
    train_mae = np.zeros((niterations, EPOCHS))
    val_mse = np.zeros((niterations, EPOCHS))
    val_mae = np.zeros((niterations, EPOCHS))
    test_mse = np.zeros((niterations, 1))
    test_mae = np.zeros((niterations, 1))
    preds_val = np.zeros((total_sentences_val,niterations))
    preds_test = np.zeros((total_sentences_test, niterations))


    for layer_size in cnn_layer_size:
        #if layer_size == 7:
            #cnn_kernel_size_on = cnn_kernel_size_mod
        #else:
        #    cnn_kernel_size_on = cnn_kernel_size
        for kernel_size in cnn_kernel_size:
            #if kernel_size == (16,8):
            #    max_pool_kernel_on = max_pool_kernel_mod
            #else:
            #    max_pool_kernel_on = max_pool_kernel
            #for npool in max_pool_kernel_on:
                #for nstride in stride_kernel:
            for fc_1_neurons in first_fc_neurons:
               for fc_2_neurons in second_fc_neurons:
                    for act_fun in act_func:
                        for dropout_val in dropout_values:
                      #              for pool_max in max_pool_bool:
                      #                  for i in range(0,2):


                                            #file_write_dir = '/home/al5517/ML_data/directional/simulated/CNN_testsim/network_data/'




                                            # The lower the batch-size, the less data the algorithm has to work with. The larger the batch size, the longer it takes to learn.

                                            # Shuffling is important to make sure that the algorithm "sees" different batches of data in each epoch.

                                            # Epochs: The algorithm will iterate over the same training dataset for several epochs, ideally until convergence is reached.
                                            #EPOCHS = 20
                                            # Different activations, like relu, softplus, have different properties with regards to convergence. Relu is generally a safe choice. Use softmax as output of a classifier.
                                            #H_ACTIVATION = 'relu'
                                            #ACTIVATION = custom_activation
                                            # Learning rate: Set this by monitoring the loss function during training.
                                            #0.001#0.0001
                                            # A dataset needs to be split into a training and test dataset.
                                            #   - Training: Part of the labelled dataset, known prior to deployment. This is used by algorithm to learn weights. This is typically a large-scale dataset that was carefully hand-annotated and released to the public (e.g., LOCATA)
                                            #   - Test: "Blind", (possibly) unlabelled dataset observed during deployment. This could be, e.g., live audio data.
                                            #TRAIN_SIZE = 0.8
                                            #TEST_SIZE = 0.2

                                            # Loss function: Target to be optimized. In general, use categorical cross-entropy for classification problems.


                                            # Load data from mat file to tfrecord (dataset format in tensorflow)


                                            # Load tfrecord datasets from file:
                                            # Process each frequency bin individually, i.e., one MLP per frequency band. For purposes of demonstration, we only use frequency bin with index 5 in the following.
                                            # Reason: A large number of frequency bins do not contain sufficient energy for meaningful results due to sparsity of speech.
                                            #epoch = 3


                                            # Build model:
                                            #model = build_model(nfeatures, OPTIMIZER, LOSS, METRICS, H_ACTIVATION)
                                            model = build_model(OPTIMIZER, LOSS, METRICS, layer_size, kernel_size, nfeatures, nb_frames, act_fun, dropout_val, fc_1_neurons, fc_2_neurons) #used to be 36, 3,3
                                            model.summary()
                                            params = model.count_params()
                                            # Train model:
                                            # Specify steps_per_epoch and validation_steps if the dataset is infinitely repeating (see load_data):
                                            #callback = EarlyStopping(monitor='val_loss', patience=2)
                                            hist = model.fit(train_dataset, validation_data=val_dataset, validation_steps=total_sentences_val//BATCH_SIZE_VAL, epochs=EPOCHS, verbose=1, steps_per_epoch= total_sentences_train//BATCH_SIZE)

                                            train_loss_results = hist.history
                                            train_mse = train_loss_results['loss']
                                            train_mae = train_loss_results['mean_absolute_error']
                                            val_mse = train_loss_results['val_loss']
                                            val_mae = train_loss_results['val_mean_absolute_error']
                                            #if len(train_loss_results['loss']) < niterations:
                                            #    train_mse[iteration, :] = np.concatenate((np.array(train_loss_results['loss']),np.ones((1,niterations-len(train_loss_results['loss'])))*train_loss_results['loss'][-1]),axis=None)
                                            #    train_mae[iteration, :] = np.concatenate((np.array(train_loss_results['mean_absolute_error']),np.ones((1,niterations-len(train_loss_results['mean_absolute_error'])))*train_loss_results['mean_absolute_error'][-1]), axis=None)
                                            #    val_mse[iteration, :] = np.concatenate((np.array(train_loss_results['val_loss']), np.ones((1,niterations - len(train_loss_results['val_loss']))) *train_loss_results['val_loss'][-1]),axis=None)
                                            #    val_mae[iteration, :] = np.concatenate((np.array(train_loss_results['val_mean_absolute_error']), np.ones((1,niterations - len(train_loss_results['val_mean_absolute_error']))) *train_loss_results['val_mean_absolute_error'][-1]), axis=None)
                                            #else:
                                            #    train_mse[iteration, :] = train_loss_results['loss']
                                            #    train_mae[iteration, :] = train_loss_results['mean_absolute_error']
                                            #    val_mse[iteration, :] = train_loss_results['val_loss']
                                            #    val_mae[iteration, :] = train_loss_results['val_mean_absolute_error']
                                            # train_loss_results = hist.history
                                            # #print(train_loss_results.dtype)
                                            # #print(train_loss_results)
                                            # train_mse = train_loss_results['loss']
                                            # #print(train_mse)
                                            # #print(type(train_mse))
                                            # train_mae = train_loss_results['mean_absolute_error']
                                            # #print(train_mae)
                                            # #print(type(train_mae))

                                                # Test:
                                            results = model.evaluate(test_dataset, steps=total_sentences_test // BATCH_SIZE)
                                            # print('test loss, test acc:', results)
                                            # print(type(results))
                                            test_mse = results[0]
                                            test_mae = results[1]
                                            print(f'test_mae: {test_mae}')
                                            print(f'test_mse: {test_mse}')

                                            preds_val = np.reshape(model.predict(X_val), (total_sentences_val))
                                            preds_test = np.reshape(model.predict(X_test), (total_sentences_test))

                                            #model.save_weights(f'{data_dir}{iteration}/model_weights_2fold')




                                            del results
                                            del hist
                                            del model
                                            del params
                                            gc.collect()
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
                                                'labels_test': y_test
                                            }
                                            if not(os.path.isdir(f'{data_dir}{iteration}/')):
                                                os.makedirs(f'{data_dir}{iteration}/')
                                            sio.savemat(f'{data_dir}{iteration}/predictions_data.mat',
                                                        results_data)
                                            iteration = iteration + 1
                                            #if not os.path.exists(file_write_dir + str(epoch) +'/'):
                                            #    os.makedirs(file_write_dir + str(epoch) +'/')
                                            #sio.savemat(file_write_dir + str(epoch) +'/' + 'results_data.mat',
                                            #            results_data)
                                            #sio.savemat(f'/home/al5517/NewDataset/model_sep/redo/{scenario}/output/{freq_idx}/predictions_data.mat',
                                            #            prediction_data)
