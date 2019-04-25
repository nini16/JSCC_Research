from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
# set_random_seed(2)

from keras.models import Model
from keras.layers import Dense, GaussianNoise, Input, LSTM, Bidirectional, Reshape, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import numpy as np

# Creation of custom AWGN layer
from keras import backend as K
from keras.engine.topology import Layer

import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


# **Model Architecture

def BuildModel_BLSTM(std):
	# set_random_seed(2)
	# placeholder for input signal
	input_sig = Input(shape=(7,1))

	# source-coding
	encode = Bidirectional(LSTM(32, return_sequences=True))(input_sig) #use bidirectional

	# channel-coding
	encode = TimeDistributed(Dense(1))(encode) #try time distributed

	#Reshape
	encode = Reshape((7,))(encode)

	encode = Dense(6, activation='linear')(encode)

	# batch-normalization
	encode = BatchNormalization(epsilon=1e-06, center=False, scale=False)(encode)

	# noise layer
	channel = AWGN( stddev=std )(encode)
	# use keras reshape layer
	# channel-decoding
	#decode = Dense(8, activation='softmax')(channel)

	decode = Dense(7, activation='relu')(channel)

	#Reshape
	decode = Reshape((7,1))(decode)

	decode = Bidirectional(LSTM(32, return_sequences=True))(decode)

	decode = TimeDistributed(Dense(1, activation='sigmoid'))(decode)

	decode = Reshape((7, ))(decode)

	model = Model(input_sig, decode)

def Plot_Acc(history, i):
	f1 = plt.figure()
	plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], 'r', label="Testing Error")
	plt.plot(range(len(history.history['loss'])), history.history['loss'], 'b', label="Training Error")
	plt.title('Train and Test Error (SNR = ' + str(SNR[i]) + 'dB)')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.xscale('linear')
	plt.yscale('linear')
	plt.grid(True)
	plt.legend()
	f1.savefig('TestTrain_' + str(SNR[i]) + '.png', format='png')


def Load_Model_Weights_BLSTM(testModel, snr):
    fName = 'MSE_model_' + str(snr) + '.h5'
    testModel.compile(optimizer='adam',
                    loss='mse',
                    metrics=['categorical_accuracy'])
    testModel.load_weights(fName)


# p1, p2, d1, p3, d2, d3, d4
codewords = [[-1,-1,-1,-1,-1,-1],
             [-1,1,-1,1,-1,1],
             [1,-1,-1,1,1,-1],
             [1,1,-1,-1,1,1],
             [1,1,1,-1,-1,-1],
             [1,-1,1,1,-1,1],
             [-1,1,1,1,1,-1],
             [-1,-1,1,-1,1,1]]


def HardDecode(arr):
	narr = []
	for element in arr:
	    if element > 0:
	        narr.append(1)
	    else:
	        narr.append(-1)
	return narr

# soft 7,4 hamming code
# vesion can be 1 for 8 bit output or 2 for 7 bit
def Hamming_Soft(inputs, stdev, signal, version=1):
	if len(np.shape(inputs)) > 2:
		inputs = np.reshape(inputs, (-1, np.shape(inputs)[1]))

	predicted = []
	for msg in inputs:
            
	    code = codewords[signal.index(msg)]
	    encoded = code + stdev * np.random.randn(np.shape(code)[0]) # encoding
	    distances = [np.linalg.norm(np.array(encoded)-np.array(cword)) for cword in codewords]
	    if version==1:
        	predicted.append(target[np.argmin(distances)])
	    else:
        	predicted.append(signal[np.argmin(distances)])
	return np.array(predicted)

def Hamming_Hard(inputs, stdev, signal, version=1):
	if len(np.shape(inputs)) > 2:
		inputs = np.reshape(inputs, (-1, np.shape(inputs)[1]))

	predicted = []
	for msg in inputs:
	    code = codewords[signal.index(msg)]
	    encoded = code + stdev * np.random.randn(np.shape(code)[0]) # encoding
	    encoded = HardDecode(encoded)
	    distances = [np.linalg.norm(np.array(encoded)-np.array(cword)) for cword in codewords]
	    if version==1:
        	predicted.append(target[np.argmin(distances)])
	    else:
        	predicted.append(signal[np.argmin(distances)])
	return np.array(predicted)


# hamming distances
hamming_distances = [[0,1,1,1,1,1,1,1],
                     [1,0,2,2,2,2,2,2],
                     [1,2,0,2,2,2,2,2],
                     [1,2,2,0,2,2,2,2],
                     [1,2,2,2,0,2,2,2],
                     [1,2,2,2,2,0,2,2],
                     [1,2,2,2,2,2,0,2],
                     [1,2,2,2,2,2,2,0]]

# bit error rate
def BER(actual, predicted):
	if len(actual) != len(predicted):
	    print "BER error: Lengths do not match"
	    return
		
	if len(np.shape(predicted)) > 2:
		predicted = np.reshape(predicted, (-1, np.shape(predicted)[1]))
		
	actSignalIndex = [target.index(act) for act in actual]
	predSignalIndex = [[1,2,3,4,5,6,7,0][np.argmax(pred)] for pred in predicted]

	return np.mean([hamming_distances[i][j] for i,j in zip(actSignalIndex, predSignalIndex)]) / 7.0
    
    
# block error rate
def BLER(actual, predicted):
	if len(actual) != len(predicted):
	    print "BER error: Lengths do not match"
	    return

	if len(np.shape(predicted)) > 2:
		predicted = np.reshape(predicted, (-1, np.shape(predicted)[1]))

	return np.mean([(np.argmax(act) != np.argmax(pred)) for act,pred in zip(actual, predicted)])

def MSE(actual, predicted):
	if len(actual) != len(predicted):
	    print "BER error: Lengths do not match"
	    return

	return np.mean([np.linalg.norm(np.array(act)-np.array(pred)) for act,pred in zip(actual, predicted)])
    


##############################################  DEMO  ########################################################

os.chdir(sys.path[0])


# Test model from -1.0 dB to 10.0 dB

# SNR = np.arange(-1.0, 11.0, 0.5)
SNR = np.arange(-1.0, 1.0, 1)
stdRange = [( 6.0/(14*(10**(snr/10))) )**.5 for snr in SNR] 


# Dataset genetation
signal =  [[0,0,0,0,0,0,0],
           [1,0,0,0,0,0,0],
           [0,1,0,0,0,0,0],
           [0,0,1,0,0,0,0],
           [0,0,0,1,0,0,0],
           [0,0,0,0,1,0,0],
           [0,0,0,0,0,1,0],
           [0,0,0,0,0,0,1]]

target =  [[0,0,0,0,0,0,0,1],
           [1,0,0,0,0,0,0,0],
           [0,1,0,0,0,0,0,0],
           [0,0,1,0,0,0,0,0],
           [0,0,0,1,0,0,0,0],
           [0,0,0,0,1,0,0,0],
           [0,0,0,0,0,1,0,0],
           [0,0,0,0,0,0,1,0]] 

indexes = np.random.choice(8, 50000)

DemoX = [signal[i] for i in indexes]
DemoX_Copy = np.array([signal[i] for i in indexes])
Demoy = [target[i] for i in indexes]

DemoX = np.reshape( DemoX, (np.shape(DemoX)+(1,)) )
# Demoy = np.reshape( Demoy, (np.shape(Demoy)[0], 1, np.shape(Demoy)[1]) )



###########################   BLSTM    #############################################3
# test with DNN jscc model trained at 0.0 dB

os.chdir(sys.path[0])

if not os.path.exists('MSE_Models'):
    os.makedirs('MSE_Models')
os.chdir('MSE_Models')

# create channels at each SNR for our jscc model
jsccModels = [BuildModel_BLSTM(std) for std in stdRange]

BLSTM_MSE_00 = []

for i in range(len(jsccModels)):
    Load_Model_Weights_BLSTM(jsccModels[i], 0.0)
    DemoPred = jsccModels[i].predict(DemoX)
    BLSTM_MSE_00.append(MSE(Demoy, DemoPred))
    
    print "Simulation on " + str(SNR[i]) + " complete!"


###########################   HAMMING code    #############################################3

MSE_hamming = []

for i in range(len(stdRange)):
    DemoPred = Hamming_Soft(DemoX_Copy, stdRange[i], signal)
    MSE_hamming.append(MSE(Demoy, DemoPred))
    
    print "Simulation on " + str(SNR[i]) + " complete!"


# test with Hamming code (hard decision)

MSE_hamming_hard = []

for i in range(len(stdRange)):
    DemoPred = Hamming_Hard(DemoX_Copy, stdRange[i], signal)
    MSE_hamming_hard.append(MSE(Demoy, DemoPred))
    
    print "Simulation on " + str(SNR[i]) + " complete!"


	
os.chdir(sys.path[0])
# plot Bit Error results

f1 = plt.figure()
# plt.plot(SNR, DNN_BitError_00, 'g-o', label='FCNN model (train-0.0 dB)') 
# plt.plot(SNR, DNN_BitError_80, 'r-o', label='FCNN model (train-8.0 dB)')
plt.plot(SNR, BLSTM_BitError_00, 'g-o', label='BLSTM model (train-0.0 dB)') 
# plt.plot(SNR, BLSTM_BitError_80, 'c-o', label='BLSTM model (train-8.0 dB)')
plt.plot(SNR, BitError_hamming, 'b-o', label='Hamming code + Soft decoder')
plt.plot(SNR, BitError_hamming_hard, 'y-o', label='Hamming code + Hard decoder')

plt.axis([-2, 10.5, 3e-5, 1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True)
# plt.title('Bit Error Rate vs SNR')
plt.legend()
# plt.show()
f1.savefig('testMSEsim.png', format='png')
