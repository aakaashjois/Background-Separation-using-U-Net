from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

def __get_conv_layer_with_dropout__(filter_size, input_layer, dropout, mask_shape = (3, 3), dropout_rate = 0.2):
    conv2d = Conv2D(filter_size, mask_shape, padding = 'same', activation = 'relu')(input_layer)
    if dropout:
        return Dropout(dropout_rate)(conv2d)
    else:
        return conv2d

def lr_schedule(epoch=1, lr = 1.5e-4):
    return lr/(epoch*0.1 + 1)    

def dice_coef(y_true, y_pred):
    y_true = K.round(K.reshape(y_true, [-1]))
    y_pred = K.round(K.reshape(y_pred, [-1]))
    isct = K.sum(y_true * y_pred)
    return 2 * isct / (K.sum(y_true) + K.sum(y_pred))

def __down__(input_layer, filters, pool = True):
    conv1 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual

def __up__(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size = (2, 2), padding = "same")(upsample)
    concat = Concatenate(axis = 3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(conv1)
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
    u1 = __up__(d5, residual = residuals[-1], filters = filters / 2)

    filters /= 2

    # __up__ 2,  32
    u2 = __up__(u1, residual = residuals[-2], filters = filters / 2)

    filters /= 2

    # __up__ 3, 64
    u3 = __up__(u2, residual = residuals[-3], filters = filters / 2)

    filters /= 2

    # __up__ 4, 128
    u4 = __up__(u3, residual = residuals[-4], filters = filters / 2)

    out = Conv2D(filters = 1, kernel_size = (1, 1), activation = 'hard_sigmoid')(u4)

    model = Model(input_layer, out)
    return model
