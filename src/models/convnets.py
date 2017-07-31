from keras.models import Sequential, Model
from keras.layers import InputLayer, Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers

def ConvolutionalNet(vocabulary_size, embedding_dimension, input_length, embedding_weights=None):
    
    model = Sequential()

    main_input = InputLayer(input_shape=(20,))
    model.add(main_input)

    if embedding_weights is None:
        model.add(Embedding(vocabulary_size, embedding_dimension, trainable=False, input_length=20))

    else:
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length,
                            weights=[embedding_weights], trainable=False, input_shape=(20, 1)))

    model.add(Convolution1D(32, 2, kernel_regularizer=regularizers.l2(0.05)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Convolution1D(32, 2, kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Convolution1D(32, 2, kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling1D(17))
    model.add(Flatten())

    model.add(Dense(1, use_bias=True, kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    return model
