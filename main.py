import os
import datetime
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split

DEBUG = False
RETRAIN = False
output_name = 'relu3layer.h5'

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
    with open('predictions_'+timestamp+'.csv', 'w') as oFile:
        oFile.write("ImageId,Label\n")
        i = 1
        for p in predictions:
            oFile.write(str(i) + "," + str(p) + "\n")
            i += 1


if not RETRAIN and os.path.isfile('models/'+output_name):
    model = load_model('models/'+output_name)
    predict(model, X_test)
    exit()

# Creating the new model
model = Sequential()
model.add(Flatten(input_shape=(28,28,1)))
# Dense = fully connected
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
# categorical_crossentropy = log-loss
model.compile(optimizer=RMSprop(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create a generator to pass the data to the model
gen = image.ImageDataGenerator()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)

# Creating the callback to fix the lr at a plateau in loss value
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

# Fitting the model
history = model.fit_generator(batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches,
                              validation_steps=val_batches.n, callbacks=[reduce_lr], verbose=1)
model.save('models/'+output_name, overwrite=True)

results = history.history
print "Training loss: ", results['loss']
print "Training accuracy: ", results['acc']
print "Validation loss: ", results['val_loss']
print "Validation accuracy: ", results['val_acc']


predict(model, X_test)
