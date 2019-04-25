import pickle, os, sys
import numpy as np
from keras.layers import Conv2D, Dense, Input, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

# Creation of custom AWGN layer
from keras import backend as K
from keras.engine.topology import Layer

from keras.layers.advanced_activations import PReLU
##class PRELU(PReLU):
##    def __init__(self, **kwargs):
##        self.__name__ = "PRELU"
##        super(PRELU, self).__init__(**kwargs)

class AWGN(Layer):

    def __init__(self, stddev, **kwargs):
        super(AWGN, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(AWGN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id=1, test_batch=""):
    if not (test_batch == ""):
        fname = cifar10_dataset_folder_path + '/' + test_batch
    else:
        fname = cifar10_dataset_folder_path + '/data_batch_' + str(batch_id)

    with open(fname, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'] / 255.0
    features = features.reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    ##features = features.reshape((len(batch['data']), 32, 32, 3))
        
    return features


TrainX = load_cfar10_batch('cifar-10-batches-py', 1)

for i in range(2,6):
    TrainX = np.vstack((TrainX, load_cfar10_batch('cifar-10-batches-py', i)))

Trainy = TrainX
TestX = Testy = load_cfar10_batch('cifar-10-batches-py', test_batch='test_batch')
print( np.shape(TrainX))

def BuildModel(std, compression_ratio):
    length = int(3072 * compression_ratio / 64)
    input_sig = Input( shape=np.shape(TrainX)[1:] )
    encode = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same') (input_sig)
    p = PReLU()
    p.__name__ = 'prelu'
    encode = p (encode)
    encode = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same') (encode)
    p = PReLU()
    p.__name__ = 'prelu'
    encode = p (encode)
    encode = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same') (encode)
    p = PReLU()
    p.__name__ = 'prelu'
    encode = p (encode)
    encode = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same') (encode)
    p = PReLU()
    p.__name__ = 'prelu'
    encode = p (encode)
    encode = Conv2D(length, kernel_size=(5, 5), strides=(1, 1), padding='same') (encode)
    p = PReLU()
    p.__name__ = 'prelu'
    encode = p (encode)
    
    encode = BatchNormalization(epsilon=1e-06, center=False, scale=False)(encode)
    channel = AWGN( stddev=std )(encode)
    
    decode = Conv2DTranspose(32, kernel_size=(5, 5), strides=(1, 1), padding='same') (channel)
    p = PReLU()
    p.__name__ = 'prelu'
    decode = p (decode)
    decode = Conv2DTranspose(32, kernel_size=(5, 5), strides=(1, 1), padding='same') (decode)
    p = PReLU()
    p.__name__ = 'prelu'
    decode = p (decode)
    decode = Conv2DTranspose(32, kernel_size=(5, 5), strides=(1, 1), padding='same') (decode)
    p = PReLU()
    p.__name__ = 'prelu'
    decode = p (decode)
    decode = Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), padding='same') (decode)
    p = PReLU()
    p.__name__ = 'prelu'
    decode = p (decode)
    decode = Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), activation='sigmoid', padding='same') (decode)
        
    model = Model(input_sig, decode)
    print(model.summary())
    return model


SNR = [8.0]
stdRange = [( 1.0/(10**(snr/10)) )**.5 for snr in SNR]
jsccModels = [BuildModel(std, compression_ratio=0.5) for std in stdRange]

for i in range(len(jsccModels)):
  print ("Now compiling model with " + str(SNR[i]) + " SNR.\n")
  jsccModels[i].compile(optimizer='adam',
		  loss='mse',
		  metrics=['mse'])
  print ("Model compilation complete. Begin training....\n")

  fName = 'img_model_' + str(SNR[i]) + '.h5'
  checkpoint = ModelCheckpoint(fName, monitor='loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]

  history = jsccModels[i].fit(x = TrainX, y = Trainy, epochs = 150, 
				  verbose = 1, validation_data = (TestX, Testy), callbacks=callbacks_list, batch_size=128)
					  
  print("Training model with " + str(SNR[i]) + " SNR complete. Next.\n")
