from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np   

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

def get_unet_model(input_shape, filters = 32):
    
    input_layer = Input(shape = input_shape)
    residuals = []

    d1, res1 = __down__(input_layer, filters)
    residuals.append(res1)

    filters *= 2
    d2, res2 = __down__(d1, filters)
    residuals.append(res2)

    filters *= 2
    d3, res3 = __down__(d2, filters)
    residuals.append(res3)

    filters *= 2
    d4, res4 = __down__(d3, filters)
    residuals.append(res4)

    filters *= 2
    d5, res4 = __down__(d3, filters)
    residuals.append(res4)

    filters *= 2
    d6 = __down__(d4, filters, pool=False)

    u1 = __up__(d5, residual = residuals[-1], filters = filters / 2)

    filters /= 2
    u2 = __up__(u1, residual = residuals[-2], filters = filters / 2)

    filters /= 2
    u3 = __up__(u2, residual = residuals[-3], filters = filters / 2)

    filters /= 2
    u4 = __up__(u3, residual = residuals[-4], filters = filters / 2)
    
    filters /= 2
    u5 = __up__(u3, residual = residuals[-4], filters = filters / 2)

    out = Conv2D(filters = 1, kernel_size = (1, 1), activation = 'sigmoid')(u5)

    model = Model(input_layer, out)
    return model