from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K 


def __create_unet_model__(input_shape, filters = 64):
    """
    Creates the U-Net model.
    
    Args:
        input_shape: List. The shape of the input image
        filters: Integer. The number of filters to use in the first layer. (Default: 64)
        
    Returns:
        A Keras Model of the U-Net.
    """
    input_layer = Input(input_shape)

    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    max_pool1 = MaxPool2D()(residual1)
    filters *= 2

    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(max_pool1)
    residual2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv2)
    max_pool2 = MaxPool2D()(residual2)
    filters *= 2

    conv3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(max_pool2)
    residual3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv3)
    max_pool3 = MaxPool2D()(residual3)
    filters *= 2

    conv4 = Conv2D(filters, (3, 3), padding='same', activation='relu')(max_pool3)
    residual4 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv4)
    max_pool4 = MaxPool2D()(residual4)
    filters *= 2

    conv5 = Conv2D(filters, (3, 3), padding='same', activation='relu')(max_pool4)
    residual5 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv5)
    filters /= 2

    filters = int(filters)
    upsample6 = UpSampling2D()(residual5)
    upconv6 = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample6)
    concat6 = Concatenate(axis=3)([residual4, upconv6])
    conv6 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat6)
    conv7 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv6)

    filters /= 2
    filters = int(filters)
    upsample7 = UpSampling2D()(conv7)
    upconv7 = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample7)
    concat7 = Concatenate(axis=3)([residual3, upconv7])
    conv8 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat7)
    conv9 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv8)

    filters /= 2
    filters = int(filters)
    upsample8 = UpSampling2D()(conv9)
    upconv8 = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample8)
    concat8 = Concatenate(axis=3)([residual2, upconv8])
    conv10 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat8)
    conv11 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv10)

    filters /= 2
    filters = int(filters)
    upsample9 = UpSampling2D()(conv11)
    upconv9 = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample9)
    concat9 = Concatenate(axis=3)([residual1, upconv9])
    conv12 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat9)
    conv13 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv12)

    out = Conv2D(filters = 1, kernel_size = (1, 1), activation = 'sigmoid')(conv13)

    return Model(input_layer, out)

def dice_coef(y_true, y_pred):
    """
    Returns the Dice Coefficient of the two images passed.
    
    Args:
        y_true: Tensor. The original image.
        y_pred: Tensor. The predicted image.
    
    Returns:
        The Dice Coefficient.
    """
    y_true = K.round(K.reshape(y_true, [-1]))
    y_pred = K.round(K.reshape(y_pred, [-1]))
    isct = K.sum(y_true * y_pred)
    return 2 * isct / (K.sum(y_true) + K.sum(y_pred))

def get_unet_model(img_width):
    """
    Obtain the compiled U-Net model with the callbacks.
    
    Args:
        img_width: Integer. The width of the input image.
    
    Returns:
        A tuple of compiled U-Net model and the list of callbacks.
    """
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
