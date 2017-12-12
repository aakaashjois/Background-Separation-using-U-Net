from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K 

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

def __create_unet_model__(input_shape, filters = 64):
    
    input_layer = Input(input_shape)
    
    d1, res1 = __down__(input_layer, filters)
    filters *= 2

    d2, res2 = __down__(d1, filters)
    filters *= 2

    d3, res3 = __down__(d2, filters)
    filters *= 2

    d4, res4 = __down__(d3, filters)
    filters *= 2

    d5 = __down__(d4, filters, pool=False)

    filters /= 2
    u1 = __up__(d5, residual = res4, filters = filters)

    filters /= 2
    u2 = __up__(u1, residual = res3, filters = filters)

    filters /= 2
    u3 = __up__(u2, residual = res2, filters = filters)

    filters /= 2
    u4 = __up__(u3, residual = res1, filters = filters)

    out = Conv2D(filters = 1, kernel_size = (1, 1), activation = 'sigmoid')(u4)

    model = Model(input_layer, out)
    return model

def get_unet_model(img_width):
    model = __create_unet_model__([img_width, img_width, 3])
    model.compile(optimizer=Adam(1e-4), 
                  loss='binary_crossentropy',
                  metrics=[dice_coef])
    checkpoint = ModelCheckpoint('best-fit.h5',
                                 monitor='val_dice_coef',
                                 save_best_only=True, 
                                 mode='max')
    callbacks = [checkpoint]
    return model, callbacks