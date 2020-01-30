from keras.layers import Input, concatenate
from keras.models import Model
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D

def my_model(dim_in, Ncl):
    
    input = Input(shape=(dim_in,dim_in,dim_in,1))
    
    x    = Conv3D(32, (3,3,3), padding='same', activation='relu')(input)
    high = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(high)
    
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    mid = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(mid)
    
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(64, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, mid])
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(48, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, high])
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    output = Conv3D(Ncl, (1,1,1), padding='same', activation='softmax')(x)
    
    model = Model(input, output)
    return model