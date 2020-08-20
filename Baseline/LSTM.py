import os
import numpy as np
from keras import layers
from keras import Model
from keras import Input
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import mplcyberpunk

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

temp = float_data[:, 1]
#a = float_data[:, 0]
#a = np.reshape(a, newshape=(len(a), 1))
#b = float_data[:, 2:]
#print(a.shape, b.shape)
#float_data = np.concatenate([a, b], axis=1)
num_features = float_data.shape[1]
print(float_data.shape, 'is the new shape of float_data')
print(temp.shape, 'is the shape of target temp')
print('*'*100)


# Axis0: time steps
# Axis1: features

# Define the mode of training pairs
# lookback, step, delay
lookback = 720
step = 6
delay = 144
batch_size = 128
# Using past five days with sampling every hr to predict the T 24 hrs later

# Split the data into training and testing
train_base = float_data[:200000]
test_base = float_data[200000:]

mean = train_base.mean(axis=0)
float_data -= mean
std = train_base.std(axis=0)
float_data /= std

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
        samples = np.zeros(shape=(len(pick_index), lookback//step, data.shape[-1]))
        targets = np.zeros(shape=(len(pick_index), ))

        for j, centroid in enumerate(pick_index):
            indices = range(centroid-lookback, centroid, step)
            samples[j] = data[indices]
            targets[j] = temp[centroid+delay]

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


# Here is a useless toy API func model for code debug
single_input = Input(shape=(lookback//step, num_features), name='input')
lstm_output = layers.LSTM(32, activation='relu', name='lstm')(single_input)
dense_output = layers.Dense(1, activation='sigmoid', name='dense')(lstm_output)

model = Model(single_input, dense_output)

print(model.summary())
print('*'*100)

model.compile(optimizer=RMSprop(), loss='mae')
hist = model.fit_generator(train_generator,
                           steps_per_epoch=train_steps,
                           epochs=20,
                           validation_data=val_generator,
                           validation_steps=val_steps)

hist_dict = hist.history
loss = hist_dict['loss']
val_loss = hist_dict['val_loss']
epochs = range(1, 1+len(val_loss))

plt.style.use('cyberpunk')
plt.plot(epochs, loss, marker='*', label='Training mae')
plt.plot(epochs, val_loss, marker='*', label='Validation mae')
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.title('Training and Validaiton Mae')
plt.legend()
mplcyberpunk.add_glow_effects()
plt.show()
plt.clf()


