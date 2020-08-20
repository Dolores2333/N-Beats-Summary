# ******************** Import a Lot *********************
import os
import numpy as np
from keras import layers
from keras import Model
from keras import Input
import keras.backend as K
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
import mplcyberpunk

#******************* Global Variables *********************
# Define the mode of training pairs
# lookback, step, delay
lookback = 720
step = 6
delay = 144
batch_size = 128
learning_rate = 1e-5
# Using past five days with sampling every hr to predict the T 24 hrs later

num_hidden=128
dim_theta=4
num_blocks=2
backcast_length = lookback // step
forecast_length = 1

# ******************** Prepare the Data *********************
# Multivariate time series
data_dir='C:/Users/11785/Desktop/PreUST/Meeting/KerasLearn/Chapter6/jena_climate'
jena_dir = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(jena_dir)
data = f.read()
f.close()
print('Type of data is:',type(data))

# Split the data into lines by '\n'
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print('Type of header is:',type(data))
print(header)
num_samples = len(lines)
num_features = len(header)-1
print(num_features, 'Original attributes in the data')
print(num_samples, 'Original time steps in the data')
print('*'*100)
# Which axis records the timestep? How many features are there? How many samples in total?

# Switch the data to float_data
float_data = np.zeros(shape=(num_samples, num_features))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i] =  values

temp_original = float_data[:, 1]
#a = float_data[:, 0]
#a = np.reshape(a, newshape=(len(a), 1))
#b = float_data[:, 2:]
#print(a.shape, b.shape)
#float_data = np.concatenate([a, b], axis=1)
num_features = float_data.shape[1]
print(float_data.shape, 'is the new shape of float_data')
print(temp_original.shape, 'is the shape of origianl target temp')
print('*'*100)


# Axis0: time steps
# Axis1: features

# Split the data into training and testing
train_base = float_data[:200000]
test_base = float_data[200000:]

mean = train_base.mean(axis=0)
float_data -= mean
std = train_base.std(axis=0)
float_data /= std

#******************** Define generators *********************
# Generate a batch of pairs of (samples, targets) at once
def generator(data, min_index, max_index,batch_size, lookback, step, delay, shuffle=False):
    if max_index is None:
        max_index = len(data)-1-delay
    # The first centroid
    i = min_index + lookback

    while 1:
        if shuffle:
            pick_index = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i+batch_size>=max_index:
                i = min_index+lookback

            pick_index = np.arange(i, min(i+batch_size, max_index))
            i+=len(pick_index)

        # pick_index stores the indexes of centroids we want to expand into training pairs
        # Once creating a pick_index, we got a batch later on
        samples = np.zeros(shape=(len(pick_index), backcast_length, data.shape[-1]))
        targets = np.zeros(shape=(len(pick_index), backcast_length+forecast_length))

        for j, centroid in enumerate(pick_index):
            indices = range(centroid-lookback, centroid, step)
            samples[j] = data[indices]
            forecast_target = temp_original[centroid+delay]
            forecast_target = np.reshape(forecast_target, newshape=(forecast_length,))
            backcast_target = temp_original[indices]
            targets[j] = np.concatenate([backcast_target, forecast_target], axis=0)

        yield samples, targets

# Instantiate the generator

train_generator = generator(data=float_data,
                            min_index=0,
                            max_index=200000,
                            batch_size=batch_size,
                            lookback=lookback,
                            step=step,
                            delay=delay,
                            shuffle=False)

val_generator = generator(data=float_data,
                          min_index=200001,
                          max_index=300000,
                          batch_size=batch_size,
                          lookback=lookback,
                          step=step,
                          delay=delay,
                          shuffle=False)

test_generator = generator(data=float_data,
                           min_index=300001,
                           max_index=None,
                           batch_size=batch_size,
                           lookback=lookback,
                           step=step,
                           delay=delay,
                           shuffle=False)

train_steps = (200000-lookback) // batch_size
val_steps = (300000-200001-lookback) // batch_size
test_steps = (num_samples- 300001-lookback) // batch_size


# ******************** Define the Network Architecture *********************
#******************* Dfine the block *********************
def mean_tensor(x):
    mean_x = K.mean(x, axis=1, keepdims=False)
    return mean_x

# Remember to change the generator tomorrow!!!! We need both back_cast and fore_cast !!!!


block_input = Input(batch_shape=(batch_size, backcast_length, num_features), name='block_input')
# Change the input shape after changing the generator!!!!Maybe
fc1 = layers.Dense(num_hidden, activation='relu', name='fc1')(block_input)
fc2 = layers.Dense(num_hidden, activation='relu', name='fc2')(fc1)
fc3 = layers.Dense(num_hidden, activation='relu', name='fc3')(fc2)
fc4 = layers.Dense(num_hidden, activation='relu', name='fc4')(fc3)
fc4_mean = layers.Lambda(mean_tensor, output_shape=(num_hidden, ))(fc4)
theta_b = layers.Dense(dim_theta, activation='linear', use_bias=False, name='theta_b')(fc4_mean)
theta_f = layers.Dense(dim_theta, activation='linear', use_bias=False, name='theta_f')(fc4_mean)
backcast = layers.Dense(backcast_length, activation='linear', name='backcast')(theta_b)
forecast = layers.Dense(forecast_length, activation='linear', name='forecast')(theta_f)
cast=layers.Concatenate(axis=-1, name='cast')([backcast, forecast])

block = Model(block_input, cast)
block.compile(optimizer=RMSprop(lr=learning_rate), loss='mae')

print(block.summary())
print('*'*100)

# Choose the real model
model = block
hist = model.fit_generator(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=20,
                    validation_data=val_generator,
                    validation_steps=val_steps)

#******************* Visualization *********************
hist_dict = hist.history
loss = hist_dict['loss']
val_loss = hist_dict['val_loss']
epochs = range(1, 1+len(loss))

plt.style.use('cyberpunk')
plt.plot(epochs, loss, marker='*', label='Training loss')
plt.plot(epochs, val_loss, marker='*', label='Validaiton loss')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
mplcyberpunk.add_glow_effects()

plt.show()
plt.clf()
