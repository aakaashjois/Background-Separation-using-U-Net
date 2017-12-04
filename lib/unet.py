from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

def __get_conv_layer_with_dropout__(filter_size, input_layer, dropout, mask_shape=(3, 3), dropout_rate=0.2):
    conv2d = Conv2D(filter_size, mask_shape, padding='same', activation='relu')(input_layer)
    if dropout:
        return Dropout(dropout_rate)(conv2d)
    else:
        return conv2d

def dice_coef(y_true, y_pred, smooth=1e-5):
    y_true = K.round(K.reshape(y_true, [-1]))
    y_pred = K.round(K.reshape(y_pred, [-1]))
    isct = K.sum(y_true * y_pred)
    return 2 * isct / (K.sum(y_true) + K.sum(y_pred))

def __down__(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual

def __up__(input_layer, residual, filters):
    filters=int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2

def get_unet_model(input_shape, filters = 64):
    
    input_layer = Input(shape = input_shape)
    residuals = []

    # __down__ 1, 128
    d1, res1 = __down__(input_layer, filters)
    residuals.append(res1)

    filters *= 2

    # __down__ 2, 64
    d2, res2 = __down__(d1, filters)
    residuals.append(res2)

    filters *= 2

    # __down__ 3, 32
    d3, res3 = __down__(d2, filters)
    residuals.append(res3)

    filters *= 2

    # __down__ 4, 16
    d4, res4 = __down__(d3, filters)
    residuals.append(res4)

    filters *= 2

    # __down__ 5, 8
    d5 = __down__(d4, filters, pool=False)

    # __up__ 1, 16
    u1 = __up__(d5, residual=residuals[-1], filters=filters/2)

    filters /= 2

    # __up__ 2,  32
    u2 = __up__(u1, residual=residuals[-2], filters=filters/2)

    filters /= 2

    # __up__ 3, 64
    u3 = __up__(u2, residual=residuals[-3], filters=filters/2)

    filters /= 2

    # __up__ 4, 128
    u4 = __up__(u3, residual=residuals[-4], filters=filters/2)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(u4)

    model = Model(input_layer, out)
    return model
    
'''    
def get_unet_model(input_shape, learning_rate, dropout=True, loss='binary_crossentropy'):
    # Input layer
    input_layer = Input(shape=input_shape)
    # ____down____ 1
    ____down_____conv_1 = __get_conv_layer_with_dropout__(64, input_layer, dropout)
    residual_1 = __get_conv_layer_with_dropout__(64, ____down_____conv_1, dropout=False)
    maxpool_1 = MaxPooling2D()(residual_1)
    # __down__ 2
    __down___conv_2 = __get_conv_layer_with_dropout__(128, maxpool_1, dropout)
    residual_2 = __get_conv_layer_with_dropout__(128, __down___conv_2, dropout=False)
    maxpool_2 = MaxPooling2D()(residual_2)
    # __down__ 3
    __down___conv_3 = __get_conv_layer_with_dropout__(256, maxpool_2, dropout)
    residual_3 = __get_conv_layer_with_dropout__(256, __down___conv_3, dropout=False)
    maxpool_3 = MaxPooling2D()(residual_3)
    # __down__ 4
    __down___conv_4 = __get_conv_layer_with_dropout__(512, maxpool_3, dropout)
    residual_4 = __get_conv_layer_with_dropout__(512, __down___conv_4, dropout=False)
    maxpool_4 = MaxPooling2D()(residual_4)
    # Flat
    flat_conv = __get_conv_layer_with_dropout__(1024, maxpool_4, dropout)
    residual_flat = __get_conv_layer_with_dropout__(1024, flat_conv, dropout=False)
    # __up__ 1
    __up__sample_1 = __up__Sampling2D()(residual_flat)
    concat_1 = Concatenate(axis=3)([residual_4, __up__sample_1])
    __up___conv_1d = __get_conv_layer_with_dropout__(512, concat_1, dropout)
    __up___conv_1 = __get_conv_layer_with_dropout__(512, __up___conv_1d, dropout=False)
    # __up__ 2
    __up__sample_2 = __up__Sampling2D()(__up___conv_1)
    concat_2 = Concatenate(axis=3)([residual_3, __up__sample_2])
    __up___conv_2d = __get_conv_layer_with_dropout__(256, concat_2, dropout)
    __up___conv_2 = __get_conv_layer_with_dropout__(256, __up___conv_2d, dropout=False)
    # __up__ 3
    __up__sample_3 = __up__Sampling2D()(__up___conv_2)
    concat_3 = Concatenate(axis=3)([residual_2, __up__sample_3])
    __up___conv_3d = __get_conv_layer_with_dropout__(128, concat_3, dropout)
    __up___conv_3 = __get_conv_layer_with_dropout__(128, __up___conv_3d, dropout=False)
    # __up__ 4
    __up__sample_4 = __up__Sampling2D()(__up___conv_3)
    concat_4 = Concatenate(axis=3)([residual_1, __up__sample_4])
    __up___conv_4d = __get_conv_layer_with_dropout__(64, concat_4, dropout)
    __up___conv_4 = __get_conv_layer_with_dropout__(64, __up___conv_4d, dropout=False)
    # Output layer
    output_layer = out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(__up___conv_4)

    # Create Model
    model = Model(input_layer, output_layer)
    model.compile(optimizer=Adam(learning_rate), loss=loss, metrics=[dice_coef])
    return model
'''