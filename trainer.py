
import keras
from keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

# x_train = np.load("data/all_input.npy").astype(np.float32)
# y_train = np.load("data/all_output.npy").astype(np.float32)
# print ('x_train', x_train.shape) #320000 x 128 x 128: 128 -- size of maps, 3200000: number data points

# np.save("data/small_input.npy", x_train[:300,:,:])
# np.save("data/small_output.npy", y_train[:300,:,:])

x_train = np.load("data/small_input.npy").astype(np.float32)
y_train = np.load("data/small_output.npy").astype(np.float32)

n_train = 200
N = x_train.shape[1]
x_train = x_train[:n_train, :, :]
y_train = y_train[:n_train, :, :]



x = Input(shape=(N, N, 1))

net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(x)
net = BatchNormalization()(net)
for i in range(19):
	net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='relu')(net)
	net = BatchNormalization()(net)
	
net = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", kernel_initializer='orthogonal', activation='sigmoid')(net)
net = BatchNormalization()(net)	
net = Dropout(0.10)(net)

model = Model(inputs=x,outputs=net)
model.summary()

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
# save_weights = ModelCheckpoint(filepath='weights_2d.hf5', monitor='val_acc',verbose=1, save_best_only=True)

print('Train network ...')
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(x_train.reshape(n_train,N,N,1), y_train.reshape(n_train,N,N,1), batch_size=32, validation_split=1/14, epochs=100, verbose=1)
print('Save trained model ...')
model.save("model_2d.hf5")




