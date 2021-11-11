import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.layers.advanced_activations import LeakyReLU

def custom_activation(x):
    return (tf.keras.backend.sigmoid(x) * 1.065) - 0.065

#def build_model(nfeatures, optimizer, loss, metrics, h_activation):
def build_model(optimizer, loss, metrics, KernelSize1, StrideConv, KernelSizePool,input_dim1, input_dim2):
    # Input size: Unelss explicitly defined (input_shape), the output size of the previous layer is used as the input size.
    # Output size: Defined by the number of units.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(3, KernelSize1, strides=StrideConv, activation='selu', input_shape=(input_dim1, input_dim2, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling2D(
        pool_size=KernelSizePool, strides=None, padding='valid', data_format=None))
    #if pool_max == True:
    #    model.add(tf.keras.layers.MaxPool2D(
    #        pool_size=npool, strides=None, padding='valid', data_format=None)
    #    )
    #elif pool_max == False:
    #    model.add(tf.keras.layers.AveragePooling2D(
    #        pool_size=npool, strides=None, padding='valid', data_format=None)
    #    )

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='selu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='selu'))
    #model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    #model.add(tf.keras.layers.Dropout(.5))
    #model.add(tf.keras.layers.Dense(128, activation='selu'))
    #model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation=custom_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
