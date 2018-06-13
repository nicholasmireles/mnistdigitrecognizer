import sys
import os
import datetime
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten,Dropout
from keras.optimizers import RMSprop,Adam,Adadelta
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau,TensorBoard
from keras.layers import BatchNormalization, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from keras import regularizers

#from helpers import plot_images

DEBUG = False
RETRAIN = True
output_name = 'CNN.h5'
batch_size = 128
epochs = 12

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def standardize(matrix):
    return (matrix-np.average(matrix))/np.std(matrix)

# Getting the features for each example
X_train = train.ix[:, 1:].values.astype('float32')
X_train = standardize(X_train)

# Getting the labels for each example
y_train = train.ix[:, 0].values.astype('int32')

# Getting test features
X_test = test.values.astype('float32')
X_test = standardize(X_test)

# Reshaping the data to (num_images,img_rows,img_cols,gray_channel) format
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Turning the Y_labels into a one-hot vector
y_train = to_categorical(y_train)
num_classes = y_train.shape[1]

# Fixing the random seed for repeatability
seed = 43
np.random.seed(seed)


def predict(trained_model, test_set):
    predictions = trained_model.predict_classes(test_set)
    timestamp=datetime.datetime.now().strftime("%Y%m%dT%H%M")
    with open('predictions/'+timestamp+'.csv', 'w') as oFile:
        oFile.write("ImageId,Label\n")
        i = 1
        for p in predictions:
            oFile.write(str(i) + "," + str(p) + "\n")
            i += 1


if not RETRAIN and os.path.isfile('models/'+output_name):
    model = load_model('models/'+output_name)
    predict(model, X_test)
    sys.exit()

# Creating the new model
model = Sequential()

#Set up some important variables
filter_size1 = 3          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36
dense_size = 128
pool_size = (2,2)

#Adding the layers
model.add(Conv2D(num_filters1,filter_size1,activation="relu",use_bias=True,input_shape=(28,28,1)))
#model.add(MaxPool2D(pool_size=pool_size,strides=(1,1),padding="same"))
model.add(Conv2D(num_filters2,filter_size2,activation="relu",use_bias=True))
model.add(MaxPool2D(pool_size=pool_size,strides=(1,1),padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(dense_size,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10,activation="softmax"))

# categorical_crossentropy = log-loss
model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

# Create a generator to pass the data to the model
#gen = image.ImageDataGenerator()

# Creating the train/validation splits
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

# Creating the callback to fix the lr at a plateau in loss value
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

# Creating a callback for tensorboard to visualize the training process
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=64,
#                          write_graph=True, write_grads=True, write_images=True, embeddings_freq=0,
#                          embeddings_layer_names=None, embeddings_metadata=None)

# Fitting the model
history = model.fit(X_train, y_train,batch_size=batch_size, epochs=epochs, validation_data=(X_val,y_val),callbacks=[reduce_lr], verbose=1)

# Saving the model
model.save('models/'+output_name, overwrite=True)

predict(model,X_test)
