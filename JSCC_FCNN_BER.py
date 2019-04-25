
# # Training


from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
# set_random_seed(2)



from keras.models import Model
from keras.layers import Dense, GaussianNoise, Input
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import numpy as np
from keras.callbacks import ModelCheckpoint


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Creation of custom AWGN layer
from keras import backend as K
from keras.engine.topology import Layer

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

indexes = np.random.choice(8, 100000)

TrainX = np.array([signal[i] for i in indexes[0:80000]])
Trainy = np.array([target[i] for i in indexes[0:80000]])

TestX = np.array([signal[i] for i in indexes[80000:100000]])
Testy = np.array([target[i] for i in indexes[80000:100000]])


# k*Ec = n*Eb 
# 
# k = 7, n = 6
# Eb = 1     {due to batch-normalization}
# 
# hence, Ec = 6/7
# 
# SNR = 10∗log(6/7(No))
# where No = 2σ^2     {σ is standard deviation}
# 
# ∴  σ^2 = 6/( 14*10^(SNR/10) )
# 
# ∴  σ = √6/( 14∗10^(SNR/10) )

# In[5]:


# SNR = np.arange(0.0, 8.5, 0.5)
SNR = [0.0, 2.5, 8.0]
stdRange = [( 6.0/(14*(10**(snr/10))) )**.5 for snr in SNR] 


# **Model Architecture**

def BuildModel(std):
	#   set_random_seed(2)
	# placeholder for input signal
	input_sig = Input(shape=(7,))

	# source-coding
	encode = Dense(3, activation='relu')(input_sig)
	# channel-coding
	encode = Dense(6, activation='relu')(encode)
	# batch-normalization
	encode = BatchNormalization(epsilon=1e-06, center=False, scale=False)(encode)

	# noise layer
	channel = AWGN( stddev=std )(encode)

	# channel-decoding
	decode = Dense(6, activation='relu')(channel)
	# source-decoding
	decode = Dense(3, activation='relu')(decode)
	# output layer
	decode = Dense(8, activation='softmax')(decode)

	model = Model(input_sig, decode)
	return model



jsccModels = [BuildModel(std) for std in stdRange]


# **Train model**


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


import os
if not os.path.exists('jscc_models_legacy'):
    os.mkdir('jscc_models_legacy')
os.chdir('jscc_models_legacy')



for i in range(len(jsccModels)):
	print ("Now compiling model with " + str(SNR[i]) + " SNR.\n")
	jsccModels[i].compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['categorical_accuracy'])
	print ("Model compilation complete. Begin training....\n")

	fName = 'legacy_model_' + str(SNR[i]) + '.h5'
	checkpoint = ModelCheckpoint(fName, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	history = jsccModels[i].fit(x = TrainX, y = Trainy, epochs = 50, 
					  verbose = 1, validation_data = (TestX, Testy), callbacks=callbacks_list, batch_size=40)
					  
	Plot_Acc(history, i)
	print("Training model with " + str(SNR[i]) + " SNR complete. Next.\n")



# Only use to load models
def Load_Model_Weights(testModel, snr):
    fName = 'legacy_model_' + str(snr) + '.h5'
    testModel.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
    testModel.load_weights(fName)


# Hamming code


# p1, p2, d1, p3, d2, d3, d4
codewords = [[-1,-1,-1,-1,-1,-1],
             [-1,1,-1,1,-1,1],
             [1,-1,-1,1,1,-1],
             [1,1,-1,-1,1,1],
             [1,1,1,-1,-1,-1],
             [1,-1,1,1,-1,1],
             [-1,1,1,1,1,-1],
             [-1,-1,1,-1,1,1]]


# In[9]:


def HardDecode(arr):
    narr = []
    for element in arr:
        if element > 0:
            narr.append(1)
        else:
            narr.append(-1)
    return narr

# soft 7,4 hamming code
def Hamming_Soft(inputs, stdev):
    predicted = []
    for msg in inputs:
        code = codewords[signal.index(msg)]
        encoded = code + stdev * np.random.randn(np.shape(code)[0]) # encoding
        distances = [np.linalg.norm(np.array(encoded)-np.array(cword)) for cword in codewords]
        predicted.append(target[np.argmin(distances)])
    return np.array(predicted)

def Hamming_Hard(inputs, stdev):
    predicted = []
    for msg in inputs:
        code = codewords[signal.index(msg)]
        encoded = code + stdev * np.random.randn(np.shape(code)[0]) # encoding
        encoded = HardDecode(encoded)
        distances = [np.linalg.norm(np.array(encoded)-np.array(cword)) for cword in codewords]
        predicted.append(target[np.argmin(distances)])
    return np.array(predicted)


# In[8]:


# # Testing!
# Hamming_Soft(signal[0:4], stdRange[2])


# **Evaluation Functions**

# In[10]:


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
        print ("BER error: Lengths do not match")
        return
    
    actSignalIndex = [target.index(act) for act in actual]
    predSignalIndex = [[1,2,3,4,5,6,7,0][np.argmax(pred)] for pred in predicted]
    
    return np.mean([hamming_distances[i][j] for i,j in zip(actSignalIndex, predSignalIndex)]) / 7.0
    
    
# block error rate
def BLER(actual, predicted):
    if len(actual) != len(predicted):
        print ("BER error: Lengths do not match")
        return
    
    return np.mean([(np.argmax(act) != np.argmax(pred)) for act,pred in zip(actual, predicted)])


##################### VALIDATION  #######################

# Test model from -1.0 dB to 10.0 dB

SNR = np.arange(-1.0, 10.5, 0.5)
stdRange = [( 6.0/(14*(10**(snr/10))) )**.5 for snr in SNR] 
# print stdRange



# create channels at each SNR for our jscc model

jsccModels = [BuildModel(std) for std in stdRange]


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
Demoy = [target[i] for i in indexes]


# In[15]:


# test with jsccc model trained at 2.5 dB

BitError_25 = []
BlockError_25 = []

for i in range(len(jsccModels)):
    Load_Model_Weights(jsccModels[i], 2.5)
    DemoPred = jsccModels[i].predict(DemoX)
    BitError_25.append(BER(Demoy, DemoPred))
    BlockError_25.append(BLER(Demoy, DemoPred))
    
    print ("Simulation on " + str(SNR[i]) + " complete!")


# In[16]:


# create channels at each SNR for our jscc model

jsccModels = [BuildModel(std) for std in stdRange]


# In[17]:


# test with jsccc model trained at 8.0 dB

BitError_80 = []
BlockError_80 = []

for i in range(len(jsccModels)):
    Load_Model_Weights(jsccModels[i], 8.0)
    DemoPred = jsccModels[i].predict(DemoX)
    BitError_80.append(BER(Demoy, DemoPred))
    BlockError_80.append(BLER(Demoy, DemoPred))
    
    print ("Simulation on " + str(SNR[i]) + " complete!")


# In[18]:


# test with Hamming code

BitError_hamming = []
BlockError_hamming = []

for i in range(len(stdRange)):
    DemoPred = Hamming_Soft(DemoX, stdRange[i])
    BitError_hamming.append(BER(Demoy, DemoPred))
    BlockError_hamming.append(BLER(Demoy, DemoPred))
    
    print ("Simulation on " + str(SNR[i]) + " complete!")


# In[19]:


# test with Hamming code (hard decision)

BitError_hamming_hard = []
BlockError_hamming_hard = []

for i in range(len(stdRange)):
    DemoPred = Hamming_Hard(DemoX, stdRange[i])
    BitError_hamming_hard.append(BER(Demoy, DemoPred))
    BlockError_hamming_hard.append(BLER(Demoy, DemoPred))
    
    print "Simulation on " + str(SNR[i]) + " complete!"


# In[23]:

# plot Bit Error results

f1 = plt.figure()
plt.plot(SNR, BitError_hamming, 'b-o', label='Soft Hamming code')
plt.plot(SNR, BitError_25, 'g-o', label='JSCC model (train-2.5 dB)') 
plt.plot(SNR, BitError_80, 'r-o', label='JSCC model (train-8.0 dB)')
plt.plot(SNR, BitError_hamming_hard, 'y-o', label='Hard Hamming code')

plt.axis([-2, 10.5, 3e-5, 1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True)
plt.title('Bit Error Rate vs SNR')
plt.legend()
f1.savefig('BERsim.png', format='png')


# In[24]:


# plot Block Error results

f1 = plt.figure()
plt.plot(SNR, BlockError_hamming, 'b-o', label='Soft Hamming code')
plt.plot(SNR, BlockError_25, 'g-o', label='JSCC model (train-2.5 dB)') 
plt.plot(SNR, BlockError_80, 'r-o', label='JSCC model (train-8.0 dB)')
plt.plot(SNR, BlockError_hamming_hard, 'y-o', label='Hard Hamming code')

plt.axis([-2, 10.5, 3e-5, 1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('Block Error Rate')
plt.grid(True)
plt.title('Block Error Rate vs SNR')
plt.legend()
f1.savefig('BLERsim.png', format='png')

