from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from keras.models import Model


def __get_conv_layer_with_dropout__(filter_size, input_layer, dropout, mask_shape=(3, 3), dropout_rate=0.2):
    conv2d = Conv2D(filter_size, mask_shape, padding='same', activation='relu')(input_layer)
    if dropout:
        return Dropout(dropout_rate)(conv2d)
    else:
        return conv2d

def dice_coef(y_true, y_pred):
    intersect = 2 * y_true * y_pred
    len_true = sqrt(sum((y_true)**2))
    len_pred = sqrt(sum((y_pred)**2))
    loss = intersect/(len_true+len_pred)
    return loss
    
def get_unet_model(input_shape, dropout=True):
    # Input layer
    input_layer = Input(shape=input_shape)
    # Down 1
    down_conv_1 = __get_conv_layer_with_dropout__(64, input_layer, dropout)
    residual_1 = __get_conv_layer_with_dropout__(64, down_conv_1, dropout=False)
    maxpool_1 = MaxPooling2D()(residual_1)
    # Down 2
    down_conv_2 = __get_conv_layer_with_dropout__(128, maxpool_1, dropout)
    residual_2 = __get_conv_layer_with_dropout__(128, down_conv_2, dropout=False)
    maxpool_2 = MaxPooling2D()(residual_2)
    # Down 3
    down_conv_3 = __get_conv_layer_with_dropout__(256, maxpool_2, dropout)
    residual_3 = __get_conv_layer_with_dropout__(256, down_conv_3, dropout=False)
    maxpool_3 = MaxPooling2D()(residual_3)
    # Down 4
    down_conv_4 = __get_conv_layer_with_dropout__(512, maxpool_3, dropout)
    residual_4 = __get_conv_layer_with_dropout__(512, down_conv_4, dropout=False)
    maxpool_4 = MaxPooling2D()(residual_4)
    # Flat
    flat_conv = __get_conv_layer_with_dropout__(1024, maxpool_4, dropout)
    residual_flat = __get_conv_layer_with_dropout__(1024, flat_conv, dropout=False)
    # Up 1
    upsample_1 = UpSampling2D()(residual_flat)
    concat_1 = Concatenate(axis=3)([residual_4, upsample_1])
    up_conv_1d = __get_conv_layer_with_dropout__(512, concat_1, dropout)
    up_conv_1 = __get_conv_layer_with_dropout__(512, up_conv_1d, dropout=False)
    # Up 2
    upsample_2 = UpSampling2D()(up_conv_1)
    concat_2 = Concatenate(axis=3)([residual_3, upsample_2])
    up_conv_2d = __get_conv_layer_with_dropout__(256, concat_2, dropout)
    up_conv_2 = __get_conv_layer_with_dropout__(256, up_conv_2d, dropout=False)
    # Up 3
    upsample_3 = UpSampling2D()(up_conv_2)
    concat_3 = Concatenate(axis=3)([residual_2, upsample_3])
    up_conv_3d = __get_conv_layer_with_dropout__(128, concat_3, dropout)
    up_conv_3 = __get_conv_layer_with_dropout__(128, up_conv_3d, dropout=False)
    # Up 4
    upsample_4 = UpSampling2D()(up_conv_3)
    concat_4 = Concatenate(axis=3)([residual_1, upsample_4])
    up_conv_4d = __get_conv_layer_with_dropout__(64, concat_4, dropout)
    up_conv_4 = __get_conv_layer_with_dropout__(64, up_conv_4d, dropout=False)
    # Output layer
    output_layer = out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up_conv_4)

    # Create Model
    model = Model(input_layer, output_layer)
    return model