import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.layers.advanced_activations import LeakyReLU

def custom_activation(x):
    return (tf.keras.backend.sigmoid(x) * 1.065) - 0.065

#def build_model(nfeatures, optimizer, loss, metrics, h_activation):
def build_model(optimizer, loss, metrics, nFilters1, KernelSize1, input_dim1, input_dim2, act_func,dropout_val, fc_1_neurons, fc_2_neurons, fc_3_neurons, nb_fc):
    # Input size: Unelss explicitly defined (input_shape), the output size of the previous layer is used as the input size.
    # Output size: Defined by the number of units.
    model = tf.keras.Sequential()
    if act_func == 'tanh':
        model.add(tf.keras.layers.Conv2D(nFilters1, KernelSize1, strides=(1,1), activation='tanh', input_shape=(input_dim1, input_dim2, 2)))
    elif act_func == 'leaky0.2':
        model.add(tf.keras.layers.Conv2D(nFilters1, KernelSize1, strides=(1, 1), activation=None,
                                         input_shape=(input_dim1, input_dim2, 2)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    #model.add(tf.keras.layers.Conv2D(nFilters1, KernelSize1, strides=(1,1), activation='tanh', input_shape=(input_dim1, input_dim2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    if nFilters1 > 1:
        model.add(tf.keras.layers.AveragePooling2D(
            pool_size = (1, nFilters1), strides=None, padding='valid', data_format=None)
        )
    #if pool_max == True:
    #    model.add(tf.keras.layers.MaxPool2D(
    #        pool_size=npool, strides=None, padding='valid', data_format=None)
    #    )
    #elif pool_max == False:
    #    model.add(tf.keras.layers.AveragePooling2D(
    #        pool_size=npool, strides=None, padding='valid', data_format=None)
    #    )

    model.add(tf.keras.layers.Flatten())
    if nb_fc > 0:
        model.add(tf.keras.layers.Dense(fc_1_neurons, activation=None))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(.5))
    if nb_fc > 1:
        model.add(tf.keras.layers.Dense(fc_2_neurons, activation=None))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization())
    if nb_fc > 2:
        model.add(tf.keras.layers.Dense(fc_3_neurons, activation=None))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_val))
    model.add(tf.keras.layers.Dense(1, activation=custom_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model